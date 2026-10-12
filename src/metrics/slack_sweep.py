"""Run seeded slack and forecast-error sweeps over scheduling policies.

All experiments use the project's generated synthetic traces, not measured
grid-carbon or electricity-price data.
"""
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jobs.job import JobGenerator
from scheduler.policies import (
    cheapest_first_policy,
    cheapest_wait_policy,
    cleanest_first_policy,
    cleanest_wait_policy,
    fifo_ondemand_policy,
    joint_policy,
    joint_wait_policy,
)
from simulator.simulator import Simulator


POLICIES = {
    "FIFO": fifo_ondemand_policy,
    "Cheapest": cheapest_first_policy,
    "Cleanest": cleanest_first_policy,
    "Joint": joint_policy,
    "Cheapest-Wait": cheapest_wait_policy,
    "Cleanest-Wait": cleanest_wait_policy,
    "Joint-Wait": joint_wait_policy,
}
SLACK_SWEEP_COLUMNS = [
    "slack",
    "policy",
    "total_cost",
    "total_carbon_g",
    "avg_wait_h",
    "deadline_success_pct",
    "cost_vs_fifo_pct",
    "carbon_vs_fifo_pct",
]
FORECAST_ERROR_COLUMNS = [
    "forecast_error_std",
    "cost_vs_fifo_pct_mean",
    "cost_vs_fifo_pct_std",
    "carbon_vs_fifo_pct_mean",
    "carbon_vs_fifo_pct_std",
]


def _project_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    return (
        root,
        root / "data" / "carbon_traces" / "sample_carbon.csv",
        root / "data" / "price_traces" / "sample_price.csv",
    )


def _simulate(
    jobs,
    policy,
    carbon_path: Path,
    price_path: Path,
    forecast_error_std: float = 0.0,
    forecast_seed: int = 0,
) -> dict:
    simulator = Simulator(
        carbon_csv_path=str(carbon_path),
        price_csv_path=str(price_path),
        forecast_horizon_hours=24,
        forecast_error_std=forecast_error_std,
        forecast_seed=forecast_seed,
    )
    results = simulator.run(jobs, policy)
    scheduled = results[results["region_used"].notna()]
    return {
        "total_cost": float(results["cost_incurred"].sum()),
        "total_carbon_g": float(results["carbon_emitted_g"].sum()),
        "avg_wait_h": float(scheduled["wait_hours"].mean()),
        "deadline_success_pct": float(100 * results["met_deadline"].mean()),
    }


def run_slack_sweep(
    output_csv: Path | str | None = None,
    output_png: Path | str | None = None,
    slacks=(1, 2, 4, 8, 12, 24),
    job_seeds=(42,),
    num_jobs: int = 60,
    window_hours: int = 120,
) -> pd.DataFrame:
    """Run all policies across fixed-slack workloads and save CSV and chart."""
    root, carbon_path, price_path = _project_paths()
    output_csv = Path(output_csv) if output_csv else root / "docs" / "slack_sweep.csv"
    output_png = Path(output_png) if output_png else root / "docs" / "slack_sweep.png"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for slack in slacks:
        for job_seed in job_seeds:
            jobs = JobGenerator.generate(
                num_jobs=num_jobs,
                window_hours=window_hours,
                min_slack_hours=slack,
                max_slack_hours=slack,
                seed=job_seed,
            )
            metrics_by_policy = {
                name: _simulate(jobs, policy, carbon_path, price_path)
                for name, policy in POLICIES.items()
            }
            fifo = metrics_by_policy["FIFO"]
            for policy_name, metrics in metrics_by_policy.items():
                rows.append({
                    "slack": slack,
                    "policy": policy_name,
                    "total_cost": metrics["total_cost"],
                    "total_carbon_g": metrics["total_carbon_g"],
                    "avg_wait_h": metrics["avg_wait_h"],
                    "deadline_success_pct": metrics["deadline_success_pct"],
                    "cost_vs_fifo_pct": _percent_change(
                        metrics["total_cost"], fifo["total_cost"]
                    ),
                    "carbon_vs_fifo_pct": _percent_change(
                        metrics["total_carbon_g"], fifo["total_carbon_g"]
                    ),
                })

    result = pd.DataFrame(rows, columns=SLACK_SWEEP_COLUMNS)
    if len(job_seeds) > 1:
        result = result.groupby(
            ["slack", "policy"], as_index=False
        )[SLACK_SWEEP_COLUMNS[2:]].mean()
        result = result[SLACK_SWEEP_COLUMNS]
    result.to_csv(output_csv, index=False, float_format="%.6f")
    _plot_slack_sweep(result, output_png)
    return result


def run_forecast_error_sweep(
    output_csv: Path | str | None = None,
    output_png: Path | str | None = None,
    forecast_error_stds=(0, 0.1, 0.2, 0.4),
    seeds=range(5),
    num_jobs: int = 60,
    window_hours: int = 120,
    slack: int = 12,
) -> pd.DataFrame:
    """Measure joint-wait cost/carbon changes over paired seed replicates."""
    root, carbon_path, price_path = _project_paths()
    output_csv = (
        Path(output_csv) if output_csv else root / "docs" / "forecast_error.csv"
    )
    output_png = (
        Path(output_png) if output_png else root / "docs" / "forecast_error.png"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    samples = {std: {"cost": [], "carbon": []} for std in forecast_error_stds}
    for seed in seeds:
        jobs = JobGenerator.generate(
            num_jobs=num_jobs,
            window_hours=window_hours,
            min_slack_hours=slack,
            max_slack_hours=slack,
            seed=seed,
        )
        fifo = _simulate(jobs, fifo_ondemand_policy, carbon_path, price_path)
        for std in forecast_error_stds:
            forecast = _simulate(
                jobs,
                joint_wait_policy,
                carbon_path,
                price_path,
                forecast_error_std=std,
                forecast_seed=seed + 10_000,
            )
            samples[std]["cost"].append(
                _percent_change(forecast["total_cost"], fifo["total_cost"])
            )
            samples[std]["carbon"].append(
                _percent_change(forecast["total_carbon_g"], fifo["total_carbon_g"])
            )

    rows = []
    for std in forecast_error_stds:
        rows.append({
            "forecast_error_std": std,
            "cost_vs_fifo_pct_mean": float(np.mean(samples[std]["cost"])),
            "cost_vs_fifo_pct_std": float(np.std(samples[std]["cost"], ddof=0)),
            "carbon_vs_fifo_pct_mean": float(np.mean(samples[std]["carbon"])),
            "carbon_vs_fifo_pct_std": float(np.std(samples[std]["carbon"], ddof=0)),
        })
    result = pd.DataFrame(rows, columns=FORECAST_ERROR_COLUMNS)
    result.to_csv(output_csv, index=False, float_format="%.6f")
    _plot_forecast_error(result, output_png)
    return result


def _percent_change(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if value == 0 else float("inf")
    return 100 * (value - baseline) / baseline


def _plot_slack_sweep(result: pd.DataFrame, output_png: Path) -> None:
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, column, title, ylabel in (
        (axes[0], "cost_vs_fifo_pct", "Cost vs FIFO", "Change (%)"),
        (axes[1], "carbon_vs_fifo_pct", "Carbon vs FIFO", "Change (%)"),
    ):
        for index, (policy, rows) in enumerate(result.groupby("policy", sort=False)):
            rows = rows.sort_values("slack")
            ax.plot(
                rows["slack"],
                rows[column],
                marker="o",
                label=policy,
                color=colors[index % len(colors)],
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Fixed slack (hours)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Policy impact of scheduling slack (synthetic traces)")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_forecast_error(result: pd.DataFrame, output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        result["forecast_error_std"],
        result["cost_vs_fifo_pct_mean"],
        yerr=result["cost_vs_fifo_pct_std"],
        marker="o",
        capsize=4,
        label="Cost vs FIFO",
    )
    ax.errorbar(
        result["forecast_error_std"],
        result["carbon_vs_fifo_pct_mean"],
        yerr=result["carbon_vs_fifo_pct_std"],
        marker="o",
        capsize=4,
        label="Carbon vs FIFO",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Forecast error standard deviation")
    ax.set_ylabel("Change vs FIFO (%)")
    ax.set_title("Joint-Wait sensitivity to forecast error (12 h slack)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    slack_results = run_slack_sweep()
    forecast_results = run_forecast_error_sweep()
    print("\nSlack sweep CSV head:")
    print(slack_results.head().to_csv(index=False), end="")
    print("\nForecast-error sweep CSV head:")
    print(forecast_results.head().to_csv(index=False), end="")
