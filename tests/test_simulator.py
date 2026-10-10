import pandas as pd
import pytest

from jobs.job import Job
from simulator.simulator import Simulator


def _select_first_region(_job, _current_hour, region_options):
    del _job, _current_hour
    return region_options[0]


def _wait_forever(_job, _current_hour, _region_options):
    del _job, _current_hour, _region_options
    return "wait"


@pytest.fixture
def simulator_csvs(tmp_path):
    timestamps = pd.date_range("2024-01-01", periods=24, freq="h")
    carbon_rows = []
    price_rows = []
    for region, carbon, cpu_price, gpu_price in [
        ("region-a", 100, 0.5, 1.0),
        ("region-b", 300, 1.5, 2.0),
    ]:
        for timestamp in timestamps:
            carbon_rows.append({
                "region": region,
                "timestamp": timestamp,
                "carbon_intensity_gco2_kwh": carbon,
            })
            price_rows.append({
                "region": region,
                "timestamp": timestamp,
                "price_cpu_usd_hr": cpu_price,
                "price_gpu_usd_hr": gpu_price,
            })

    carbon_path = tmp_path / "carbon.csv"
    price_path = tmp_path / "prices.csv"
    pd.DataFrame(carbon_rows).to_csv(carbon_path, index=False)
    pd.DataFrame(price_rows).to_csv(price_path, index=False)

    return Simulator(str(carbon_path), str(price_path))


def test_cost_and_carbon_are_computed_from_duration_and_region_data(simulator_csvs):
    job = Job(
        job_id="job-cost-carbon",
        job_type="cpu_openmp",
        arrival_time=0,
        deadline=3,
        duration_estimate_hours=2,
        resource_requirement=1,
    )

    results = simulator_csvs.run([job], _select_first_region)
    result = results.iloc[0]

    assert result["region_used"] == "region-a"
    assert result["cost_incurred"] == pytest.approx(2 * 0.5)
    assert result["carbon_emitted_g"] == pytest.approx(
        Simulator.POWER_DRAW["cpu_openmp"] * 2 * 100
    )


def test_start_immediately_policy_meets_each_deadline(simulator_csvs):
    jobs = [
        Job(
            job_id="job-first",
            job_type="cpu_openmp",
            arrival_time=0,
            deadline=2,
            duration_estimate_hours=2,
            resource_requirement=1,
        ),
        Job(
            job_id="job-second",
            job_type="gpu_cuda",
            arrival_time=1,
            deadline=4,
            duration_estimate_hours=3,
            resource_requirement=1,
        ),
    ]

    results = simulator_csvs.run(
        jobs,
        _select_first_region,
    )

    assert len(results) == len(jobs)
    assert results["met_deadline"].all()


def test_wait_forever_marks_job_unscheduled_and_failed(simulator_csvs):
    job = Job(
        job_id="job-never-scheduled",
        job_type="cpu_openmp",
        arrival_time=0,
        deadline=1,
        duration_estimate_hours=1,
        resource_requirement=1,
    )

    results = simulator_csvs.run([job], _wait_forever)
    result = results.iloc[0]

    assert pd.isna(result["region_used"])
    assert pd.isna(result["start_hour"])
    assert pd.isna(result["completion_hour"])
    assert not result["met_deadline"]
    assert result["cost_incurred"] == 0
    assert result["carbon_emitted_g"] == 0
