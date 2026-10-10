"""
Job model and generator for the GreenSched simulator.
"""
from dataclasses import dataclass
from typing import List, Optional
import random
import pandas as pd


@dataclass
class Job:
    """Represents a computational job to be scheduled."""
    job_id: str
    job_type: str  # "cpu_openmp" or "gpu_cuda"
    arrival_time: float  # hour offset from simulation start
    deadline: float  # hour offset from simulation start
    duration_estimate_hours: float
    resource_requirement: int  # cpu_cores for cpu_openmp, gpu_count for gpu_cuda


class JobGenerator:
    """Generates realistic job queues for simulation."""
    
    @staticmethod
    def generate(
        num_jobs: int,
        window_hours: int,
        cpu_ratio: float = 0.5,
        min_slack_hours: float = 2,
        max_slack_hours: float = 12,
        seed: Optional[int] = None,
    ) -> List[Job]:
        """
        Generate a randomized but realistic list of Job objects.
        
        Args:
            num_jobs: Number of jobs to generate
            window_hours: Time window (hours) over which to spread arrival times
            cpu_ratio: Fraction of jobs that are CPU (vs GPU) jobs
            min_slack_hours: Minimum slack between (arrival + duration) and deadline
            max_slack_hours: Maximum slack between (arrival + duration) and deadline
            seed: Optional random seed for reproducible job generation
            
        Returns:
            List of Job objects sorted by arrival_time
        """
        rng = random.Random(seed) if seed is not None else random
        jobs = []
        
        for i in range(num_jobs):
            # Randomly determine job type
            job_type = "cpu_openmp" if rng.random() < cpu_ratio else "gpu_cuda"
            
            # Traces are hourly, so submissions are aligned to whole hours.
            arrival_time = rng.randint(0, window_hours - 1)
            
            # Duration varies by type
            if job_type == "cpu_openmp":
                duration_estimate = rng.uniform(0.5, 8)  # 30 min to 8 hours
                resource_requirement = rng.randint(1, 16)  # 1-16 CPU cores
            else:  # gpu_cuda
                duration_estimate = rng.uniform(1, 24)  # 1 to 24 hours
                resource_requirement = rng.randint(1, 4)  # 1-4 GPUs
            
            # Slack hours between (arrival + duration) and deadline
            slack = rng.uniform(min_slack_hours, max_slack_hours)
            deadline = arrival_time + duration_estimate + slack
            
            job = Job(
                job_id=f"job_{i:04d}",
                job_type=job_type,
                arrival_time=arrival_time,
                deadline=deadline,
                duration_estimate_hours=duration_estimate,
                resource_requirement=resource_requirement
            )
            jobs.append(job)
        
        # Sort by arrival time
        jobs.sort(key=lambda j: j.arrival_time)
        return jobs


if __name__ == "__main__":
    # Generate 60 jobs and print as DataFrame
    random.seed(42)
    jobs = JobGenerator.generate(num_jobs=60, window_hours=168)  # 1 week
    
    # Convert to DataFrame for display
    job_dicts = [
        {
            "job_id": j.job_id,
            "job_type": j.job_type,
            "arrival_time": round(j.arrival_time, 2),
            "deadline": round(j.deadline, 2),
            "duration_estimate_hours": round(j.duration_estimate_hours, 2),
            "resource_requirement": j.resource_requirement,
        }
        for j in jobs
    ]
    
    df = pd.DataFrame(job_dicts)
    print(f"Generated {len(df)} jobs:")
    print(df.to_string())
    print(f"\nSummary:")
    print(f"  Total jobs: {len(df)}")
    print(f"  CPU jobs: {len(df[df['job_type'] == 'cpu_openmp'])}")
    print(f"  GPU jobs: {len(df[df['job_type'] == 'gpu_cuda'])}")
    print(f"  Time window: {df['arrival_time'].min():.1f} to {df['deadline'].max():.1f} hours")
