import pandas as pd
import pytest

from jobs.job import Job
from simulator.simulator import Simulator


@pytest.fixture
def simulator_factory(tmp_path):
    timestamps = pd.date_range("2024-01-01", periods=12, freq="h")
    carbon_path = tmp_path / "carbon.csv"
    price_path = tmp_path / "prices.csv"
    pd.DataFrame({
        "region": ["region-a"] * len(timestamps),
        "timestamp": timestamps,
        "carbon_intensity_gco2_kwh": [100 * (hour + 1) for hour in range(12)],
    }).to_csv(carbon_path, index=False)
    pd.DataFrame({
        "region": ["region-a"] * len(timestamps),
        "timestamp": timestamps,
        "price_cpu_usd_hr": [hour + 1 for hour in range(12)],
        "price_gpu_usd_hr": [2 * (hour + 1) for hour in range(12)],
    }).to_csv(price_path, index=False)

    def create_simulator(**kwargs):
        return Simulator(str(carbon_path), str(price_path), **kwargs)

    return create_simulator


def _job(
    *,
    arrival_time=1,
    deadline=6,
    duration=2,
    job_type="cpu_openmp",
    resource_requirement=8,
):
    return Job(
        job_id="forecast-job",
        job_type=job_type,
        arrival_time=arrival_time,
        deadline=deadline,
        duration_estimate_hours=duration,
        resource_requirement=resource_requirement,
    )


def test_zero_error_candidates_equal_true_duration_weighted_averages(
    simulator_factory,
):
    simulator = simulator_factory(forecast_error_std=0)
    candidates = simulator.build_candidates(_job(), current_hour=1)

    assert candidates == [
        {
            "region": "region-a",
            "start_hour": start,
            "price_per_hr": pytest.approx(start + 1.5),
            "carbon_intensity": pytest.approx(100 * (start + 1.5)),
        }
        for start in range(1, 5)
    ]


def test_observed_current_hour_is_not_noised(simulator_factory):
    simulator = simulator_factory(forecast_error_std=0.8, forecast_seed=5)
    candidates = simulator.build_candidates(
        _job(duration=1, deadline=4),
        current_hour=2,
    )

    current = candidates[0]
    assert current["start_hour"] == 2
    assert current["price_per_hr"] == pytest.approx(3)
    assert current["carbon_intensity"] == pytest.approx(300)


def test_forecast_candidates_are_seeded_and_different_seeds_differ(
    simulator_factory,
):
    job = _job()
    first = simulator_factory(forecast_error_std=0.5, forecast_seed=11)
    repeated = simulator_factory(forecast_error_std=0.5, forecast_seed=11)
    different = simulator_factory(forecast_error_std=0.5, forecast_seed=12)

    first_candidates = first.build_candidates(job, current_hour=1)

    assert first_candidates == repeated.build_candidates(job, current_hour=1)
    assert first_candidates != different.build_candidates(job, current_hour=1)


def test_candidates_do_not_start_after_latest_feasible_hour(simulator_factory):
    simulator = simulator_factory(forecast_horizon_hours=24)
    job = _job(arrival_time=1, deadline=5.5, duration=2)

    candidates = simulator.build_candidates(job, current_hour=1)

    assert [candidate["start_hour"] for candidate in candidates] == [1, 2, 3]


def test_less_than_one_hour_of_slack_only_allows_current_start(simulator_factory):
    simulator = simulator_factory()
    job = _job(arrival_time=1, deadline=3.5, duration=2)

    candidates = simulator.build_candidates(job, current_hour=1)

    assert [candidate["start_hour"] for candidate in candidates] == [1]


def test_forecast_aware_policy_receives_candidates(simulator_factory):
    simulator = simulator_factory()
    received = []

    def forecast_policy(_job, _current_hour, region_options, *, forecast):
        received.extend(forecast)
        return region_options[0]

    forecast_policy.needs_forecast = True
    simulator.run([_job(duration=1, deadline=2)], forecast_policy)

    assert received
    assert all("start_hour" in candidate for candidate in received)


def test_forecast_policy_choice_is_accounted_at_its_selected_start(
    simulator_factory,
):
    simulator = simulator_factory()

    def choose_next_hour(_job, _current_hour, _region_options, *, forecast):
        return next(
            candidate for candidate in forecast if candidate["start_hour"] == 2
        )

    choose_next_hour.needs_forecast = True
    result = simulator.run(
        [_job(deadline=3, duration=1)],
        choose_next_hour,
    ).iloc[0]

    assert result["start_hour"] == 2
    assert result["wait_hours"] == 1
    assert result["cost_incurred"] == pytest.approx(1.5)
    assert result["carbon_emitted_g"] == pytest.approx(24)


def test_non_forecast_realized_accounting_ignores_forecast_error(simulator_factory):
    job = _job(duration=2, deadline=4)

    def first_region(_job, _current_hour, region_options):
        return region_options[0]

    perfect = simulator_factory(forecast_error_std=0)
    noisy = simulator_factory(forecast_error_std=0.9, forecast_seed=37)

    pd.testing.assert_frame_equal(
        perfect.run([job], first_region),
        noisy.run([job], first_region),
    )
