"""
Hour-stepped simulator for job scheduling across regions.
Executes jobs according to a scheduling policy and records outcomes.

The CPU price column is USD per 16-core instance-hour, charged pro rata
by cores. The GPU price column is USD per GPU-hour.
"""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from typing import Callable, List, Dict, Optional, Tuple


@dataclass
class ResourceModel:
    """Resource power and CPU instance sizing used for job accounting."""

    cpu_cores_per_instance: float = 16
    cpu_power_per_core_kw: float = 0.01
    gpu_power_per_gpu_kw: float = 0.4


class Simulator:
    """
    Simulates job scheduling across multiple regions with carbon and price data.
    """

    def __init__(
        self,
        carbon_csv_path: str,
        price_csv_path: str,
        resource_model: Optional[ResourceModel] = None,
        forecast_horizon_hours: int = 24,
        forecast_error_std: float = 0.0,
        forecast_seed: int = 0,
    ):
        """
        Initialize simulator with carbon and price data.
        
        Args:
            carbon_csv_path: Path to carbon intensity CSV
            price_csv_path: Path to spot price CSV
            resource_model: CPU instance sizing and per-resource power draw
            forecast_horizon_hours: Maximum number of hours to consider future starts
            forecast_error_std: Standard deviation of log-normal forecast error
            forecast_seed: Seed used to generate reproducible forecast errors
        """
        if forecast_horizon_hours < 0:
            raise ValueError("forecast_horizon_hours must be non-negative.")
        if forecast_error_std < 0:
            raise ValueError("forecast_error_std must be non-negative.")

        self.carbon_df = pd.read_csv(carbon_csv_path)
        self.price_df = pd.read_csv(price_csv_path)
        self.resource_model = resource_model if resource_model is not None else ResourceModel()
        self.forecast_horizon_hours = forecast_horizon_hours
        self.forecast_error_std = forecast_error_std
        
        # Parse timestamps
        self.carbon_df["timestamp"] = pd.to_datetime(self.carbon_df["timestamp"])
        self.price_df["timestamp"] = pd.to_datetime(self.price_df["timestamp"])
        
        # Create hour offset from start
        carbon_start = self.carbon_df["timestamp"].min()
        price_start = self.price_df["timestamp"].min()
        
        self.carbon_df["hour"] = (
            (self.carbon_df["timestamp"] - carbon_start).dt.total_seconds() / 3600
        ).astype(int)
        self.price_df["hour"] = (
            (self.price_df["timestamp"] - price_start).dt.total_seconds() / 3600
        ).astype(int)
        
        # Index by region and hour for fast lookup
        self.carbon_index = self.carbon_df.set_index(["region", "hour"])
        self.price_index = self.price_df.set_index(["region", "hour"])
        
        self.results = []
        self.region_names = self.carbon_df["region"].unique().tolist()
        self._build_forecast_noise(forecast_seed)

    def _build_forecast_noise(self, seed: int) -> None:
        """Create fixed multiplicative forecast errors for each trace point."""
        rng = np.random.default_rng(seed)
        price_keys = list(self.price_index.index)
        carbon_keys = list(self.carbon_index.index)
        if self.forecast_error_std == 0:
            self._price_noise = dict.fromkeys(price_keys, 1.0)
            self._carbon_noise = dict.fromkeys(carbon_keys, 1.0)
            return

        self._price_noise = dict(zip(
            price_keys,
            np.exp(rng.normal(0, self.forecast_error_std, len(price_keys))),
        ))
        self._carbon_noise = dict(zip(
            carbon_keys,
            np.exp(rng.normal(0, self.forecast_error_std, len(carbon_keys))),
        ))

    def _integrate_forecast_window(
        self,
        region: str,
        job_type: str,
        start: int,
        duration: float,
        current_hour: int,
    ) -> Tuple[float, float]:
        """Integrate forecast price and intensity over a proposed run window."""
        end = start + duration
        if duration <= 0:
            raise ValueError(f"Job duration must be positive, got {duration}.")

        trace_end = min(
            self.carbon_df.loc[self.carbon_df["region"] == region, "hour"].max() + 1,
            self.price_df.loc[self.price_df["region"] == region, "hour"].max() + 1,
        )
        if pd.isna(trace_end) or end > trace_end:
            raise ValueError(
                f"Forecast window [{start}, {end}) for region '{region}' extends "
                f"past the end of the available trace (hour {trace_end})."
            )

        price_column = (
            "price_cpu_usd_hr" if job_type == "cpu_openmp" else "price_gpu_usd_hr"
        )
        price_integral = 0.0
        intensity_integral = 0.0
        final_hour = math.ceil(end)
        for hour in range(start, final_hour):
            overlap = min(end, hour + 1) - max(start, hour)
            if overlap <= 0:
                continue
            key = (region, hour)
            try:
                price = float(self.price_index.loc[key, price_column])
                intensity = float(
                    self.carbon_index.loc[key, "carbon_intensity_gco2_kwh"]
                )
            except KeyError as error:
                raise ValueError(
                    f"Missing hourly trace data for region '{region}' at hour {hour}."
                ) from error

            if hour > current_hour:
                price *= self._price_noise[key]
                intensity *= self._carbon_noise[key]
            price_integral += overlap * price
            intensity_integral += overlap * intensity

        return price_integral / duration, intensity_integral / duration

    def build_candidates(self, job, current_hour: int) -> List[Dict]:
        """Build region/start choices with duration-weighted forecast averages."""
        duration = job.duration_estimate_hours
        latest_start = math.floor(job.deadline - duration + 1e-9)
        final_start = min(
            latest_start,
            current_hour + self.forecast_horizon_hours,
        )
        candidates = []
        for region in self.region_names:
            for start_hour in range(current_hour, final_start + 1):
                price, intensity = self._integrate_forecast_window(
                    region,
                    job.job_type,
                    start_hour,
                    duration,
                    current_hour,
                )
                candidates.append({
                    "region": region,
                    "start_hour": start_hour,
                    "price_per_hr": price,
                    "carbon_intensity": intensity,
                })
        return candidates

    def integrate_window(
        self,
        region: str,
        job_type: str,
        start: float,
        duration: float,
    ) -> Tuple[float, float]:
        """Integrate price and carbon intensity over a job's run window.

        Each hourly trace value applies to its half-open hour [k, k + 1).
        Returns (price integral, intensity integral), both weighted by
        overlapping hours but not by the job's resource quantity.
        """
        start = float(start)
        duration = float(duration)
        end = start + duration
        if start < 0:
            raise ValueError(f"Job window starts before the trace at hour {start}.")
        if duration < 0:
            raise ValueError(f"Job duration must be non-negative, got {duration}.")
        if duration == 0:
            return 0.0, 0.0

        trace_end = min(
            self.carbon_df.loc[self.carbon_df["region"] == region, "hour"].max() + 1,
            self.price_df.loc[self.price_df["region"] == region, "hour"].max() + 1,
        )
        if pd.isna(trace_end) or end > trace_end:
            raise ValueError(
                f"Job window [{start}, {end}) for region '{region}' extends "
                f"past the end of the available trace (hour {trace_end})."
            )

        price_column = (
            "price_cpu_usd_hr" if job_type == "cpu_openmp" else "price_gpu_usd_hr"
        )
        price_integral = 0.0
        intensity_integral = 0.0
        first_hour = int(start)
        final_hour = int(end) if end.is_integer() else int(end) + 1

        for hour in range(first_hour, final_hour):
            overlap = min(end, hour + 1) - max(start, hour)
            if overlap <= 0:
                continue
            try:
                price = self.price_index.loc[(region, hour), price_column]
                intensity = self.carbon_index.loc[
                    (region, hour), "carbon_intensity_gco2_kwh"
                ]
            except KeyError as error:
                raise ValueError(
                    f"Missing hourly trace data for region '{region}' at hour {hour}."
                ) from error
            price_integral += overlap * float(price)
            intensity_integral += overlap * float(intensity)

        return price_integral, intensity_integral
    
    def get_region_options(
        self,
        current_hour: int,
        job_type: str
    ) -> List[Dict]:
        """
        Get available region options for a given job type at current hour.
        
        Args:
            current_hour: Current simulation hour
            job_type: "cpu_openmp" or "gpu_cuda"
            
        Returns:
            List of dicts with region, carbon_intensity, price_per_hr
        """
        options = []
        
        for region in self.region_names:
            try:
                # Get carbon intensity
                carbon = self.carbon_index.loc[(region, current_hour), "carbon_intensity_gco2_kwh"]
                
                # Get appropriate price
                if job_type == "cpu_openmp":
                    price = self.price_index.loc[(region, current_hour), "price_cpu_usd_hr"]
                else:  # gpu_cuda
                    price = self.price_index.loc[(region, current_hour), "price_gpu_usd_hr"]
                
                options.append({
                    "region": region,
                    "carbon_intensity": float(carbon),
                    "price_per_hr": float(price),
                })
            except KeyError:
                # Region/hour combination not available
                pass
        
        return options
    
    def run(
        self,
        jobs: List,
        policy: Callable
    ) -> pd.DataFrame:
        """
        Run simulation with given job list and scheduling policy.
        
        Args:
            jobs: List of Job objects (sorted by arrival_time)
            policy: Scheduling function with signature:
                    policy(job, current_hour, region_options) -> region_dict or "wait"
                    
        Returns:
            DataFrame with job results
        """
        self.results = []
        
        # Track which jobs have been scheduled
        scheduled = {}  # job_id -> result dict
        unscheduled = {job.job_id: job for job in jobs}
        
        # Determine simulation time window
        max_deadline = max(job.deadline for job in jobs)
        max_hour = int(max_deadline) + 1
        
        # Step through simulation time
        for current_hour in range(max_hour):
            # Process unscheduled jobs that have arrived
            jobs_to_process = []
            for job_id, job in list(unscheduled.items()):
                if job.arrival_time <= current_hour:
                    jobs_to_process.append((job_id, job))
            
            # Call policy for each arrived job
            for job_id, job in jobs_to_process:
                # Get region options for this job type
                region_options = self.get_region_options(current_hour, job.job_type)
                
                if not region_options:
                    continue  # No regions available
                
                # Only forecast-aware policies receive the look-ahead candidates.
                uses_forecast = getattr(policy, "needs_forecast", False) is True
                if uses_forecast:
                    candidates = self.build_candidates(job, current_hour)
                    decision = policy(
                        job,
                        current_hour,
                        region_options,
                        forecast=candidates,
                    )
                else:
                    decision = policy(job, current_hour, region_options)
                
                if decision != "wait" and decision is not None:
                    # Schedule this job
                    region_dict = decision
                    region_name = region_dict["region"]
                    start_hour = (
                        region_dict.get("start_hour", current_hour)
                        if uses_forecast
                        else current_hour
                    )

                    price_integral, intensity_integral = self.integrate_window(
                        region_name,
                        job.job_type,
                        start_hour,
                        job.duration_estimate_hours,
                    )
                    if job.job_type == "cpu_openmp":
                        units = (
                            job.resource_requirement
                            / self.resource_model.cpu_cores_per_instance
                        )
                        power_kw = (
                            job.resource_requirement
                            * self.resource_model.cpu_power_per_core_kw
                        )
                    else:  # gpu_cuda
                        units = job.resource_requirement
                        power_kw = (
                            job.resource_requirement
                            * self.resource_model.gpu_power_per_gpu_kw
                        )

                    cost_incurred = units * price_integral
                    carbon_emitted = power_kw * intensity_integral
                    
                    # Completion hour
                    completion_hour = start_hour + job.duration_estimate_hours
                    
                    # Check deadline
                    met_deadline = (completion_hour <= job.deadline)
                    
                    result = {
                        "job_id": job.job_id,
                        "job_type": job.job_type,
                        "region_used": region_name,
                        "start_hour": start_hour,
                        "wait_hours": start_hour - job.arrival_time,
                        "completion_hour": completion_hour,
                        "met_deadline": met_deadline,
                        "cost_incurred": cost_incurred,
                        "carbon_emitted_g": carbon_emitted,
                    }
                    
                    scheduled[job_id] = result
                    del unscheduled[job_id]
        
        # Add scheduled jobs to results
        for job_id in scheduled:
            self.results.append(scheduled[job_id])
        
        # Handle any remaining unscheduled jobs (mark as failed)
        for job_id, job in unscheduled.items():
            self.results.append({
                "job_id": job_id,
                "job_type": job.job_type,
                "region_used": None,
                "start_hour": None,
                "wait_hours": None,
                "completion_hour": None,
                "met_deadline": False,
                "cost_incurred": 0.0,
                "carbon_emitted_g": 0.0,
            })
        
        results_df = pd.DataFrame(self.results)
        return results_df


if __name__ == "__main__":
    # Import Job and JobGenerator
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from jobs.job import Job, JobGenerator
    
    # Generate test jobs
    print("Generating test jobs...")
    import random
    random.seed(42)
    jobs = JobGenerator.generate(num_jobs=60, window_hours=168)
    
    # Initialize simulator
    print("Initializing simulator...")
    sim = Simulator(
        carbon_csv_path="data/carbon_traces/sample_carbon.csv",
        price_csv_path="data/price_traces/sample_price.csv",
    )
    
    # Define a simple "always pick first region immediately" policy
    def trivial_policy(job, current_hour, region_options):
        """Simple baseline: always pick first region, never wait."""
        if region_options:
            return region_options[0]
        return "wait"
    
    # Run simulation
    print("Running simulation with trivial policy...")
    results_df = sim.run(jobs, trivial_policy)
    
    # Print results
    print("\n" + "="*80)
    print("SIMULATION RESULTS")
    print("="*80)
    print(results_df.to_string())
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total jobs: {len(results_df)}")
    print(f"Jobs scheduled: {results_df['region_used'].notna().sum()}")
    print(f"Jobs met deadline: {results_df['met_deadline'].sum()}")
    print(f"Deadline success rate: {100 * results_df['met_deadline'].sum() / len(results_df):.1f}%")
    print(f"Total cost: ${results_df['cost_incurred'].sum():.2f}")
    print(f"Total carbon: {results_df['carbon_emitted_g'].sum():.0f} g CO2")
    print(f"Average cost per job: ${results_df[results_df['region_used'].notna()]['cost_incurred'].mean():.4f}")
    print(f"Average carbon per job: {results_df[results_df['region_used'].notna()]['carbon_emitted_g'].mean():.0f} g CO2")
