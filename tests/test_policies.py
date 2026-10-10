from types import SimpleNamespace

import pytest

from scheduler.policies import (
    DEFAULT_REGION,
    cheapest_first_policy,
    cleanest_first_policy,
    fifo_ondemand_policy,
    joint_policy,
)


@pytest.fixture
def region_options():
    return [
        {"region": "A", "price_per_hr": 1.0, "carbon_intensity": 100},
        {"region": "B", "price_per_hr": 0.5, "carbon_intensity": 300},
        {
            "region": DEFAULT_REGION,
            "price_per_hr": 0.7,
            "carbon_intensity": 150,
        },
    ]


@pytest.fixture
def job():
    return SimpleNamespace(deadline=10, duration_estimate_hours=1)


def test_cheapest_policy_picks_lowest_price(job, region_options):
    assert cheapest_first_policy(job, 0, region_options)["region"] == "B"


def test_cleanest_policy_picks_lowest_carbon(job, region_options):
    assert cleanest_first_policy(job, 0, region_options)["region"] == "A"


def test_joint_policy_balances_price_and_carbon(job, region_options):
    assert joint_policy(job, 0, region_options)["region"] == DEFAULT_REGION


def test_fifo_policy_prefers_default_home_region(job, region_options):
    assert fifo_ondemand_policy(job, 0, region_options)["region"] == DEFAULT_REGION


def test_fifo_policy_falls_back_to_first_option_when_home_region_is_missing(
    job,
    region_options,
):
    options_without_home = region_options[:2]

    assert fifo_ondemand_policy(job, 0, options_without_home) == options_without_home[0]


@pytest.mark.parametrize(
    ("policy", "expected_region"),
    [
        (cheapest_first_policy, "B"),
        (cleanest_first_policy, "A"),
        (joint_policy, DEFAULT_REGION),
        (fifo_ondemand_policy, DEFAULT_REGION),
    ],
)
def test_policies_use_fallback_when_no_option_meets_deadline(
    policy,
    expected_region,
    region_options,
):
    expired_job = SimpleNamespace(deadline=1, duration_estimate_hours=2)

    assert policy(expired_job, 0, region_options)["region"] == expected_region


@pytest.mark.parametrize(
    "policy",
    [
        fifo_ondemand_policy,
        cheapest_first_policy,
        cleanest_first_policy,
        joint_policy,
    ],
)
def test_policies_wait_when_no_options_are_available(policy, job):
    assert policy(job, 0, []) == "wait"
