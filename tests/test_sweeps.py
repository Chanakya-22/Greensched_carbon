import pandas as pd

from metrics.slack_sweep import (
    ABLATION_COLUMNS,
    FORECAST_ERROR_COLUMNS,
    SLACK_SWEEP_COLUMNS,
    run_ablation,
    run_forecast_error_sweep,
    run_slack_sweep,
)


def test_reduced_sweeps_write_expected_outputs_and_pass_deadlines(tmp_path):
    slack_csv = tmp_path / "slack.csv"
    slack_png = tmp_path / "slack.png"
    slack_results = run_slack_sweep(
        output_csv=slack_csv,
        output_png=slack_png,
        slacks=(1, 2),
        job_seeds=(42,),
        num_jobs=8,
        window_hours=12,
    )

    assert slack_csv.exists()
    assert slack_png.exists()
    assert slack_results.columns.tolist() == SLACK_SWEEP_COLUMNS
    assert slack_results["slack"].unique().tolist() == [1, 2]
    assert slack_results["deadline_success_pct"].eq(100).all()
    assert slack_results["policy"].nunique() == 10

    forecast_csv = tmp_path / "forecast-error.csv"
    forecast_png = tmp_path / "forecast-error.png"
    forecast_results = run_forecast_error_sweep(
        output_csv=forecast_csv,
        output_png=forecast_png,
        forecast_error_stds=(0, 0.2),
        seeds=(42,),
        num_jobs=8,
        window_hours=12,
    )

    assert forecast_csv.exists()
    assert forecast_png.exists()
    assert forecast_results.columns.tolist() == FORECAST_ERROR_COLUMNS
    assert forecast_results["forecast_error_std"].tolist() == [0, 0.2]
    assert pd.read_csv(forecast_csv).columns.tolist() == FORECAST_ERROR_COLUMNS
    assert forecast_results.loc[0, "cost_diff_vs_sigma0_pct_mean"] == 0
    assert forecast_results.loc[0, "carbon_diff_vs_sigma0_pct_mean"] == 0

    ablation_csv = tmp_path / "ablation.csv"
    ablation_png = tmp_path / "ablation.png"
    ablation_results = run_ablation(
        output_csv=ablation_csv,
        output_png=ablation_png,
        slacks=(1, 2),
        num_jobs=8,
        window_hours=12,
    )
    assert ablation_csv.exists()
    assert ablation_png.exists()
    assert ablation_results.columns.tolist() == ABLATION_COLUMNS
