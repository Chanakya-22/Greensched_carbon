from pandas.testing import assert_frame_equal

from data.synthetic_data import DEFAULT_REGIONS, generate_region_traces


EXPECTED_COLUMNS = [
    "region",
    "timestamp",
    "hour_of_day",
    "carbon_intensity_gco2_kwh",
    "price_cpu_usd_hr",
    "price_gpu_usd_hr",
]


def test_generated_trace_shape_and_values():
    combined, carbon, prices = generate_region_traces(DEFAULT_REGIONS)

    assert combined.columns.tolist() == EXPECTED_COLUMNS
    assert carbon.columns.tolist() == [
        "region",
        "timestamp",
        "carbon_intensity_gco2_kwh",
    ]
    assert prices.columns.tolist() == [
        "region",
        "timestamp",
        "price_cpu_usd_hr",
        "price_gpu_usd_hr",
    ]
    assert combined["region"].nunique() == 3
    assert combined.groupby("region").size().eq(168).all()
    assert combined["carbon_intensity_gco2_kwh"].ge(0).all()
    assert combined["price_cpu_usd_hr"].gt(0).all()
    assert combined["price_gpu_usd_hr"].gt(0).all()


def test_seed_reproduces_all_trace_dataframes():
    first_run = generate_region_traces(DEFAULT_REGIONS)
    second_run = generate_region_traces(DEFAULT_REGIONS, seed=42)

    for first, second in zip(first_run, second_run):
        assert_frame_equal(first, second)


def test_different_seeds_produce_different_traces():
    first_run = generate_region_traces(DEFAULT_REGIONS, seed=42)
    second_run = generate_region_traces(DEFAULT_REGIONS, seed=43)

    assert not first_run[0].equals(second_run[0])
