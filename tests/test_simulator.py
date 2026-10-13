import pandas as pd
import pytest

from jobs.job import Job
from simulator.simulator import ResourceModel, Simulator


def _select_first_region(_job, _current_hour, region_options):
    del _job, _current_hour
    return region_options[0]


def _wait_forever(_job, _current_hour, _region_options):
    del _job, _current_hour, _region_options
    return "wait"


def _wait_until_latest_start(job, current_hour, region_options):
    latest_start = int(job.deadline - job.duration_estimate_hours + 1e-9)
    if current_hour < latest_start:
        return "wait"
    return region_options[0]


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

    return Simulator(
        str(carbon_path),
        str(price_path),
        resource_model=ResourceModel(),
    )


@pytest.fixture
def hand_computed_simulator(tmp_path):
    timestamps = pd.date_range("2024-01-01", periods=6, freq="h")
    carbon_path = tmp_path / "hand-carbon.csv"
    price_path = tmp_path / "hand-prices.csv"
    pd.DataFrame({
        "region": ["region-a"] * 6,
        "timestamp": timestamps,
        "carbon_intensity_gco2_kwh": [100, 200, 300, 400, 500, 600],
    }).to_csv(carbon_path, index=False)
    pd.DataFrame({
        "region": ["region-a"] * 6,
        "timestamp": timestamps,
        "price_cpu_usd_hr": [1, 2, 3, 4, 5, 6],
        "price_gpu_usd_hr": [1, 2, 3, 4, 5, 6],
    }).to_csv(price_path, index=False)
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
    assert result["cost_incurred"] == pytest.approx((1 / 16) * 2 * 0.5)
    assert result["carbon_emitted_g"] == pytest.approx(1 * 0.01 * 2 * 100)
    assert result["wait_hours"] == 0


def test_cpu_window_integrates_hourly_price_and_intensity(hand_computed_simulator):
    job = Job(
        job_id="job-cpu-window",
        job_type="cpu_openmp",
        arrival_time=1,
        deadline=5,
        duration_estimate_hours=2.5,
        resource_requirement=8,
    )

    result = hand_computed_simulator.run([job], _select_first_region).iloc[0]

    assert result["cost_incurred"] == pytest.approx(3.5)
    assert result["carbon_emitted_g"] == pytest.approx(56)


def test_gpu_window_integrates_hourly_price_and_intensity(hand_computed_simulator):
    job = Job(
        job_id="job-gpu-window",
        job_type="gpu_cuda",
        arrival_time=1,
        deadline=5,
        duration_estimate_hours=2.5,
        resource_requirement=2,
    )

    result = hand_computed_simulator.run([job], _select_first_region).iloc[0]

    assert result["cost_incurred"] == pytest.approx(14)
    assert result["carbon_emitted_g"] == pytest.approx(560)


def test_job_window_past_trace_end_raises_value_error(hand_computed_simulator):
    with pytest.raises(ValueError, match="extends past the end of the available trace"):
        hand_computed_simulator.integrate_window("region-a", "cpu_openmp", 5, 2)


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


def test_deadline_rounding_tolerance_accepts_latest_feasible_start(simulator_csvs):
    duration = 0.9571817860738686
    job = Job(
        job_id="job-deadline-rounding",
        job_type="cpu_openmp",
        arrival_time=5,
        deadline=5 + duration + 5,
        duration_estimate_hours=duration,
        resource_requirement=1,
    )

    result = simulator_csvs.run([job], _wait_until_latest_start).iloc[0]

    assert result["completion_hour"] > job.deadline
    assert result["completion_hour"] - job.deadline < 1e-9
    assert result["met_deadline"]


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
    assert pd.isna(result["wait_hours"])
    assert pd.isna(result["completion_hour"])
    assert not result["met_deadline"]
    assert result["cost_incurred"] == 0
    assert result["carbon_emitted_g"] == 0
