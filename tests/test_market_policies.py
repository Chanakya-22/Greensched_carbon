from types import SimpleNamespace

import pandas as pd

from scheduler.policies import (
    market_joint,
    market_cheapest,
    make_market_policy,
    ondemand_only_fifo,
    spot_fifo,
    spot_only_joint_wait,
)
from simulator.simulator import Simulator


def _job(deadline=4, duration=1):
    return SimpleNamespace(
        deadline=deadline,
        duration_estimate_hours=duration,
        job_type="cpu_openmp",
    )


def _candidate(region, market, price, start_hour=0, carbon=100):
    return {
        "region": region,
        "start_hour": start_hour,
        "price_per_hr": price,
        "carbon_intensity": carbon,
        "market": market,
    }


def test_zero_hazard_prefers_spot_unless_spot_is_excluded_by_buffer():
    policy = make_market_policy("joint", safety_buffer_h=1.0)
    forecast = [
        _candidate("region-a", "spot", 2.0),
        _candidate("region-a", "on_demand", 2.0),
    ]

    capped_price_tie = policy(
        _job(),
        0,
        [],
        forecast=forecast,
        hazards={"region-a": 0.0},
    )
    cheaper_spot = policy(
        _job(),
        0,
        [],
        forecast=[
            _candidate("region-a", "spot", 1.0),
            _candidate("region-a", "on_demand", 2.0),
        ],
        hazards={"region-a": 0.0},
    )
    no_buffer_slack = policy(
        _job(deadline=1.5),
        0,
        [],
        forecast=forecast,
        hazards={"region-a": 0.0},
    )

    assert capped_price_tie["market"] == "spot"
    assert cheaper_spot["market"] == "spot"
    assert no_buffer_slack["market"] == "on_demand"
    assert set(no_buffer_slack) == {
        "region",
        "carbon_intensity",
        "price_per_hr",
        "market",
    }


def test_spot_fifo_uses_home_region_without_selecting_on_demand():
    options = [
        _candidate("region-cheap-dirty", "spot", 1.0),
        _candidate("region-balanced", "spot", 2.0),
        _candidate("region-balanced", "on_demand", 5.0),
    ]

    selected = spot_fifo(_job(), 0, options)

    assert selected["region"] == "region-balanced"
    assert selected["market"] == "spot"


def test_simulator_passes_hazards_and_generates_both_market_candidates(tmp_path):
    timestamps = pd.date_range("2024-01-01", periods=5, freq="h")
    carbon_path = tmp_path / "carbon.csv"
    price_path = tmp_path / "prices.csv"
    pd.DataFrame({
        "region": ["region-a"] * 5,
        "timestamp": timestamps,
        "carbon_intensity_gco2_kwh": [100] * 5,
    }).to_csv(carbon_path, index=False)
    pd.DataFrame({
        "region": ["region-a"] * 5,
        "timestamp": timestamps,
        "price_cpu_usd_hr": [1, 2, 3, 4, 5],
        "price_gpu_usd_hr": [1, 2, 3, 4, 5],
    }).to_csv(price_path, index=False)
    simulator = Simulator(
        str(carbon_path),
        str(price_path),
        include_on_demand=True,
        interruption_hazard={"region-a": 0.25},
    )

    candidates = simulator.build_candidates(_job(), current_hour=0)
    decisions = []

    def capture_hazards(job, current_hour, options, *, forecast, hazards):
        decisions.append((forecast, hazards))
        return market_joint(
            job,
            current_hour,
            options,
            forecast=forecast,
            hazards=hazards,
        )

    capture_hazards.needs_forecast = True
    capture_hazards.needs_market_info = True
    simulator.run(
        [
            SimpleNamespace(
                job_id="integration",
                job_type="cpu_openmp",
                arrival_time=0,
                deadline=4,
                duration_estimate_hours=1,
                resource_requirement=16,
            )
        ],
        capture_hazards,
    )

    assert {candidate["market"] for candidate in candidates} == {
        "spot",
        "on_demand",
    }
    assert decisions
    assert decisions[0][1] == {"region-a": 0.25}


def test_zero_hazard_cheaper_spot_wins_and_market_policy_flags_are_set():
    forecast = [
        _candidate("region-a", "spot", 1.0),
        _candidate("region-a", "on_demand", 2.0),
    ]
    decision = market_cheapest(
        _job(),
        0,
        [],
        forecast=forecast,
        hazards={"region-a": 0.0},
    )

    assert decision["market"] == "spot"
    assert market_cheapest.needs_forecast
    assert market_cheapest.needs_market_info
    assert market_cheapest.__name__ == "market_cost_policy"


def test_hazard_risk_adjustment_ranks_equal_price_spot_options():
    forecast = [
        _candidate("risky-region", "spot", 10.0),
        _candidate("safe-region", "spot", 10.0),
    ]

    decision = market_cheapest(
        _job(duration=2),
        0,
        [],
        forecast=forecast,
        hazards={"risky-region": 0.5, "safe-region": 0.0},
    )

    assert decision["region"] == "safe-region"


def test_baseline_policies_filter_to_requested_markets():
    job = _job()
    mixed_options = [
        {"region": "spot-region", "market": "spot"},
        {"region": "ondemand-region", "market": "on_demand"},
    ]

    fifo_choice = ondemand_only_fifo(job, 0, mixed_options)
    spot_choice = spot_only_joint_wait(
        job,
        0,
        mixed_options,
        forecast=[
            _candidate("spot-region", "spot", 1),
            _candidate("ondemand-region", "on_demand", 0.5),
        ],
    )

    assert fifo_choice == mixed_options[1]
    assert spot_choice["region"] == "spot-region"
