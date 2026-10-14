from pathlib import Path

import pandas as pd
import pytest

from jobs.job import Job, JobGenerator
from scheduler.policies import (
    cheapest_first_policy,
    cheapest_wait_policy,
    cheapest_window,
    cleanest_first_policy,
    cleanest_wait_policy,
    cleanest_window,
    fifo_ondemand_policy,
    joint_policy,
    joint_wait_policy,
    joint_window,
)
from simulator.simulator import Simulator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICIES = {
    "FIFO OnDemand": fifo_ondemand_policy,
    "Cheapest First": cheapest_first_policy,
    "Cleanest First": cleanest_first_policy,
    "Joint Policy": joint_policy,
    "Cheapest-Wait": cheapest_wait_policy,
    "Cleanest-Wait": cleanest_wait_policy,
    "Joint-Wait": joint_wait_policy,
    "Cheapest-Window": cheapest_window,
    "Cleanest-Window": cleanest_window,
    "Joint-Window": joint_window,
}


def test_phase2_tight_slack_cost_and_carbon_match_golden():
    carbon_path = PROJECT_ROOT / "data" / "carbon_traces" / "sample_carbon.csv"
    price_path = PROJECT_ROOT / "data" / "price_traces" / "sample_price.csv"
    golden = pd.read_csv(
        PROJECT_ROOT / "tests" / "golden" / "phase2_policy_comparison.csv"
    ).set_index("Policy")
    jobs = JobGenerator.generate(
        num_jobs=60,
        window_hours=120,
        min_slack_hours=0.5,
        max_slack_hours=3.0,
        seed=42,
    )

    actual = {}
    for policy_name, policy in POLICIES.items():
        simulator = Simulator(
            carbon_csv_path=str(carbon_path),
            price_csv_path=str(price_path),
            forecast_error_std=0.0,
            forecast_horizon_hours=24,
            forecast_seed=42,
        )
        results = simulator.run(jobs, policy)
        actual[policy_name] = {
            "Total Cost (USD)": results["cost_incurred"].sum(),
            "Total Carbon (gCO2eq)": results["carbon_emitted_g"].sum(),
        }

    assert set(actual) == set(golden.index)
    for policy_name, metrics in actual.items():
        for metric, value in metrics.items():
            assert value == pytest.approx(
                golden.loc[policy_name, metric],
                abs=1e-6,
                rel=0,
            ), f"{policy_name}: {metric} changed"


def test_on_demand_price_is_constant_and_both_markets_are_offered(tmp_path):
    timestamps = pd.date_range("2024-01-01", periods=4, freq="h")
    carbon_path = tmp_path / "carbon.csv"
    price_path = tmp_path / "prices.csv"
    pd.DataFrame({
        "region": ["region-a"] * 4,
        "timestamp": timestamps,
        "carbon_intensity_gco2_kwh": [100, 200, 300, 400],
    }).to_csv(carbon_path, index=False)
    pd.DataFrame({
        "region": ["region-a"] * 4,
        "timestamp": timestamps,
        "price_cpu_usd_hr": [1, 3, 20, 2],
        "price_gpu_usd_hr": [2, 4, 10, 4],
    }).to_csv(price_path, index=False)
    simulator = Simulator(
        str(carbon_path),
        str(price_path),
        on_demand_multiplier=2.5,
        include_on_demand=True,
    )

    expected_on_demand_price = 6.5 * 2.5
    for hour in range(4):
        options = simulator.get_region_options(hour, "cpu_openmp")
        options_by_market = {option["market"]: option for option in options}

        assert set(options_by_market) == {"spot", "on_demand"}
        assert options_by_market["on_demand"]["price_per_hr"] == pytest.approx(
            expected_on_demand_price
        )
        assert options_by_market["spot"]["price_per_hr"] <= (
            options_by_market["on_demand"]["price_per_hr"]
        )

    candidates = simulator.build_candidates(
        Job(
            job_id="market-candidate",
            job_type="cpu_openmp",
            arrival_time=0,
            deadline=2,
            duration_estimate_hours=1,
            resource_requirement=16,
        ),
        current_hour=0,
    )
    assert {candidate["market"] for candidate in candidates} == {
        "spot",
        "on_demand",
    }


def test_spot_only_options_and_accounting_remain_unmodified(tmp_path):
    timestamps = pd.date_range("2024-01-01", periods=2, freq="h")
    carbon_path = tmp_path / "carbon.csv"
    price_path = tmp_path / "prices.csv"
    pd.DataFrame({
        "region": ["region-a"] * 2,
        "timestamp": timestamps,
        "carbon_intensity_gco2_kwh": [100, 100],
    }).to_csv(carbon_path, index=False)
    pd.DataFrame({
        "region": ["region-a"] * 2,
        "timestamp": timestamps,
        "price_cpu_usd_hr": [1, 20],
        "price_gpu_usd_hr": [2, 30],
    }).to_csv(price_path, index=False)
    simulator = Simulator(str(carbon_path), str(price_path))

    options = simulator.get_region_options(1, "cpu_openmp")
    assert options == [{
        "region": "region-a",
        "carbon_intensity": 100.0,
        "price_per_hr": 20.0,
        "market": "spot",
    }]

    job = Job(
        job_id="on-demand-accounting",
        job_type="cpu_openmp",
        arrival_time=0,
        deadline=1,
        duration_estimate_hours=1,
        resource_requirement=16,
    )

    def choose_on_demand(_job, _current_hour, options):
        return next(option for option in options if option["market"] == "on_demand")

    with_on_demand = Simulator(
        str(carbon_path),
        str(price_path),
        include_on_demand=True,
    )
    result = with_on_demand.run([job], choose_on_demand).iloc[0]

    assert result["cost_incurred"] == pytest.approx(26.25)
