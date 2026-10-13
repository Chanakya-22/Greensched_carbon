import pandas as pd
import pytest

from data.synthetic_data import DEFAULT_REGIONS, generate_region_traces
from jobs.job import Job, JobGenerator
from scheduler.policies import (
    cheapest_first_policy,
    cheapest_wait_policy,
    cleanest_first_policy,
    cleanest_wait_policy,
    joint_wait_policy,
    make_wait_policy,
    cheapest_window,
    cleanest_window,
    joint_window,
)
from simulator.simulator import Simulator


@pytest.fixture
def hand_trace_simulator(tmp_path):
    timestamps = pd.date_range("2024-01-01", periods=6, freq="h")
    carbon_path = tmp_path / "carbon.csv"
    price_path = tmp_path / "prices.csv"
    pd.DataFrame({
        "region": ["region-a"] * 6,
        "timestamp": timestamps,
        "carbon_intensity_gco2_kwh": [1000, 800, 100, 200, 300, 400],
    }).to_csv(carbon_path, index=False)
    pd.DataFrame({
        "region": ["region-a"] * 6,
        "timestamp": timestamps,
        "price_cpu_usd_hr": [10, 8, 1, 2, 3, 4],
        "price_gpu_usd_hr": [10, 8, 1, 2, 3, 4],
    }).to_csv(price_path, index=False)
    return Simulator(str(carbon_path), str(price_path), forecast_error_std=0)


def _forecast_job(deadline=4):
    return Job(
        job_id="wait-job",
        job_type="cpu_openmp",
        arrival_time=0,
        deadline=deadline,
        duration_estimate_hours=1,
        resource_requirement=8,
    )


@pytest.mark.parametrize(
    "policy",
    [cheapest_wait_policy, cleanest_wait_policy, joint_wait_policy],
)
def test_wait_policies_wait_until_hour_two_for_cheaper_cleaner_forecast(
    hand_trace_simulator,
    policy,
):
    job = _forecast_job(deadline=4)
    region_options = hand_trace_simulator.get_region_options(0, job.job_type)
    forecast = hand_trace_simulator.build_candidates(job, 0)

    assert policy(job, 0, region_options, forecast=forecast) == "wait"
    result = hand_trace_simulator.run([job], policy).iloc[0]

    assert result["start_hour"] == 2


@pytest.mark.parametrize(
    "policy",
    [cheapest_wait_policy, cleanest_wait_policy, joint_wait_policy],
)
def test_wait_policies_start_at_latest_feasible_hour(
    hand_trace_simulator,
    policy,
):
    job = _forecast_job(deadline=3)
    forecast = hand_trace_simulator.build_candidates(job, 2)
    region_options = hand_trace_simulator.get_region_options(2, job.job_type)

    decision = policy(job, 2, region_options, forecast=forecast)

    assert decision != "wait"
    assert decision["region"] == "region-a"


@pytest.mark.parametrize(
    "policy",
    [cheapest_wait_policy, cleanest_wait_policy, joint_wait_policy],
)
def test_wait_policies_do_not_wait_with_less_than_one_hour_slack(
    hand_trace_simulator,
    policy,
):
    job = _forecast_job(deadline=1.5)
    forecast = hand_trace_simulator.build_candidates(job, 0)
    region_options = hand_trace_simulator.get_region_options(0, job.job_type)

    decision = policy(job, 0, region_options, forecast=forecast)

    assert decision != "wait"
    assert decision["region"] == "region-a"


@pytest.mark.parametrize(
    "policy",
    [cheapest_wait_policy, cleanest_wait_policy, joint_wait_policy],
)
def test_wait_policies_never_wait_on_flat_traces(tmp_path, policy):
    timestamps = pd.date_range("2024-01-01", periods=6, freq="h")
    carbon_path = tmp_path / "flat-carbon.csv"
    price_path = tmp_path / "flat-prices.csv"
    pd.DataFrame({
        "region": ["region-a"] * 6,
        "timestamp": timestamps,
        "carbon_intensity_gco2_kwh": [100] * 6,
    }).to_csv(carbon_path, index=False)
    pd.DataFrame({
        "region": ["region-a"] * 6,
        "timestamp": timestamps,
        "price_cpu_usd_hr": [1] * 6,
        "price_gpu_usd_hr": [1] * 6,
    }).to_csv(price_path, index=False)
    simulator = Simulator(str(carbon_path), str(price_path), forecast_error_std=0)
    job = _forecast_job(deadline=4)

    decision = policy(
        job,
        0,
        simulator.get_region_options(0, job.job_type),
        forecast=simulator.build_candidates(job, 0),
    )

    assert decision != "wait"
    assert decision["region"] == "region-a"


def test_wait_policy_has_readable_name_and_rejects_unknown_objective():
    assert make_wait_policy("cost").__name__ == "cost_wait_policy"
    assert make_wait_policy("carbon").needs_forecast
    assert make_wait_policy("joint").needs_forecast

    with pytest.raises(ValueError, match="objective must be"):
        make_wait_policy("latency")


@pytest.mark.parametrize(
    "policy",
    [cheapest_window, cleanest_window, joint_window],
)
def test_window_policies_never_wait(hand_trace_simulator, policy):
    job = _forecast_job()
    decision = policy(
        job,
        0,
        hand_trace_simulator.get_region_options(0, job.job_type),
        forecast=hand_trace_simulator.build_candidates(job, 0),
    )
    assert decision != "wait"


def test_cheapest_wait_cost_is_no_more_than_window_cost(synthetic_results):
    simulator, jobs = synthetic_results

    window = simulator.run(jobs, cheapest_window).set_index("job_id")
    waiting = simulator.run(jobs, cheapest_wait_policy).set_index("job_id")

    for job in jobs:
        assert waiting.loc[job.job_id, "cost_incurred"] <= (
            window.loc[job.job_id, "cost_incurred"] + 1e-9
        ), job.job_id


def test_cleanest_wait_carbon_is_no_more_than_window_carbon(synthetic_results):
    simulator, jobs = synthetic_results

    window = simulator.run(jobs, cleanest_window).set_index("job_id")
    waiting = simulator.run(jobs, cleanest_wait_policy).set_index("job_id")

    for job in jobs:
        assert waiting.loc[job.job_id, "carbon_emitted_g"] <= (
            window.loc[job.job_id, "carbon_emitted_g"] + 1e-9
        ), job.job_id


@pytest.fixture
def synthetic_results(tmp_path):
    _, carbon_df, price_df = generate_region_traces(
        DEFAULT_REGIONS,
        hours=168,
        seed=42,
    )
    carbon_path = tmp_path / "synthetic-carbon.csv"
    price_path = tmp_path / "synthetic-prices.csv"
    carbon_df.to_csv(carbon_path, index=False)
    price_df.to_csv(price_path, index=False)
    jobs = JobGenerator.generate(
        num_jobs=40,
        window_hours=48,
        min_slack_hours=2,
        max_slack_hours=12,
        seed=42,
    )
    simulator = Simulator(str(carbon_path), str(price_path), forecast_error_std=0)
    return simulator, jobs


def test_cheapest_wait_never_costs_more_than_cheapest_first(synthetic_results):
    simulator, jobs = synthetic_results

    first = simulator.run(jobs, cheapest_first_policy).set_index("job_id")
    wait = simulator.run(jobs, cheapest_wait_policy).set_index("job_id")

    for job in jobs:
        assert wait.loc[job.job_id, "cost_incurred"] <= (
            first.loc[job.job_id, "cost_incurred"] + 1e-9
        ), job.job_id


def test_cleanest_wait_never_emits_more_than_cleanest_first(synthetic_results):
    simulator, jobs = synthetic_results

    first = simulator.run(jobs, cleanest_first_policy).set_index("job_id")
    wait = simulator.run(jobs, cleanest_wait_policy).set_index("job_id")

    for job in jobs:
        assert wait.loc[job.job_id, "carbon_emitted_g"] <= (
            first.loc[job.job_id, "carbon_emitted_g"] + 1e-9
        ), job.job_id


def test_window_policy_raises_instead_of_waiting_without_current_candidate(
    hand_trace_simulator,
):
    policy = make_wait_policy("cost", allow_wait=False)

    with pytest.raises(ValueError, match="No current-hour forecast candidate"):
        policy(_forecast_job(), 0, [], forecast=[])
