import pandas as pd
import pytest

from jobs.job import Job
from simulator.simulator import Simulator


def _write_traces(tmp_path, hours=16):
    timestamps = pd.date_range("2024-01-01", periods=hours, freq="h")
    carbon_path = tmp_path / "carbon.csv"
    price_path = tmp_path / "prices.csv"
    pd.DataFrame({
        "region": ["region-a"] * hours,
        "timestamp": timestamps,
        "carbon_intensity_gco2_kwh": [100] * hours,
    }).to_csv(carbon_path, index=False)
    pd.DataFrame({
        "region": ["region-a"] * hours,
        "timestamp": timestamps,
        "price_cpu_usd_hr": list(range(1, hours + 1)),
        "price_gpu_usd_hr": list(range(1, hours + 1)),
    }).to_csv(price_path, index=False)
    return str(carbon_path), str(price_path)


def _spot_policy(_job, _current_hour, options):
    return next(option for option in options if option["market"] == "spot")


def test_reclaim_keeps_checkpoint_and_accounts_for_wasted_work(tmp_path):
    carbon_path, price_path = _write_traces(tmp_path)
    simulator = Simulator(
        carbon_path,
        price_path,
        checkpoint_interval_h=1.5,
        max_spot_interruptions=3,
    )
    simulator.reclaim = {
        ("region-a", hour): hour == 2
        for hour in range(16)
    }
    job = Job(
        job_id="checkpointed",
        job_type="cpu_openmp",
        arrival_time=0,
        deadline=8,
        duration_estimate_hours=3.5,
        resource_requirement=16,
    )
    decisions = []

    def record_decision(pending_job, current_hour, options):
        decisions.append((
            pending_job.arrival_time,
            pending_job.duration_estimate_hours,
            current_hour,
        ))
        return _spot_policy(pending_job, current_hour, options)

    result = simulator.run([job], record_decision).iloc[0]

    assert decisions == [(0, 3.5, 0), (2, 2.0, 3)]
    assert result["interruptions"] == 1
    assert result["wasted_hours"] == pytest.approx(0.5)
    assert result["wasted_cost"] == pytest.approx(1.0)
    assert result["wasted_carbon_g"] == pytest.approx(8.0)
    assert result["cost_incurred"] == pytest.approx(12.0)
    assert result["carbon_emitted_g"] == pytest.approx(64.0)
    assert result["completion_hour"] == pytest.approx(5.0)
    assert result["final_market"] == "spot"


def test_hazard_one_forces_on_demand_after_three_interruptions(tmp_path):
    carbon_path, price_path = _write_traces(tmp_path)
    simulator = Simulator(
        carbon_path,
        price_path,
        interruption_hazard={"region-a": 1.0},
        interruption_seed=22,
    )
    jobs = [
        Job(
            job_id=f"job-{number}",
            job_type="cpu_openmp",
            arrival_time=0,
            deadline=10,
            duration_estimate_hours=4,
            resource_requirement=16,
        )
        for number in range(3)
    ]

    results = simulator.run(jobs, _spot_policy)

    assert len(results) == len(jobs)
    assert results["job_id"].is_unique
    assert results["interruptions"].tolist() == [3, 3, 3]
    assert results["final_market"].tolist() == ["on_demand"] * 3
    assert results["completion_hour"].tolist() == pytest.approx([7, 7, 7])


def test_tight_deadline_can_be_missed_after_repeated_interruptions(tmp_path):
    carbon_path, price_path = _write_traces(tmp_path)
    simulator = Simulator(
        carbon_path,
        price_path,
        interruption_hazard={"region-a": 1.0},
    )
    jobs = [
        Job(
            job_id="tight-deadline",
            job_type="cpu_openmp",
            arrival_time=0,
            deadline=5.5,
            duration_estimate_hours=4,
            resource_requirement=16,
        )
    ]

    result = simulator.run(jobs, _spot_policy).iloc[0]

    assert result["interruptions"] == 3
    assert result["final_market"] == "on_demand"
    assert result["completion_hour"] == pytest.approx(7)
    assert not result["met_deadline"]


def test_reclaim_sampling_is_reproducible_and_seeded(tmp_path):
    carbon_path, price_path = _write_traces(tmp_path, hours=64)
    settings = {"interruption_hazard": {"region-a": 0.5}}

    first = Simulator(carbon_path, price_path, interruption_seed=17, **settings)
    repeated = Simulator(carbon_path, price_path, interruption_seed=17, **settings)
    different = Simulator(carbon_path, price_path, interruption_seed=18, **settings)

    assert first.reclaim == repeated.reclaim
    assert first.reclaim != different.reclaim
