"""
Discrete-event simulator for job scheduling across regions.
Executes jobs according to a scheduling policy and records outcomes.
"""
import pandas as pd
from typing import Callable, List, Dict, Optional
from datetime import datetime


class Simulator:
    """
    Simulates job scheduling across multiple regions with carbon and price data.
    """
    
    # Power draw estimates (kW)
    POWER_DRAW = {
        "cpu_openmp": 0.15,  # kW per CPU job
        "gpu_cuda": 0.4,      # kW per GPU job
    }
    
    def __init__(self, carbon_csv_path: str, price_csv_path: str):
        """
        Initialize simulator with carbon and price data.
        
        Args:
            carbon_csv_path: Path to carbon intensity CSV
            price_csv_path: Path to spot price CSV
        """
        self.carbon_df = pd.read_csv(carbon_csv_path)
        self.price_df = pd.read_csv(price_csv_path)
        
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
                
                # Call policy
                decision = policy(job, current_hour, region_options)
                
                if decision != "wait" and decision is not None:
                    # Schedule this job
                    region_dict = decision
                    region_name = region_dict["region"]
                    
                    # Calculate cost and carbon
                    cost_incurred = (
                        job.duration_estimate_hours * 
                        region_dict["price_per_hr"]
                    )
                    
                    # Power draw in kW
                    power_kw = self.POWER_DRAW[job.job_type]
                    
                    # Carbon: kW * hours * gCO2/kWh = gCO2
                    carbon_emitted = (
                        power_kw *
                        job.duration_estimate_hours *
                        region_dict["carbon_intensity"]
                    )
                    
                    # Completion hour
                    completion_hour = current_hour + job.duration_estimate_hours
                    
                    # Check deadline
                    met_deadline = (completion_hour <= job.deadline)
                    
                    result = {
                        "job_id": job.job_id,
                        "job_type": job.job_type,
                        "region_used": region_name,
                        "start_hour": current_hour,
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
