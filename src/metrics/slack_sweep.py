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


POLICIES = {
    "FIFO": fifo_ondemand_policy,
    "Cheapest": cheapest_first_policy,
    "Cleanest": cleanest_first_policy,
    "Joint": joint_policy,
    "Cheapest-Wait": cheapest_wait_policy,
    "Cleanest-Wait": cleanest_wait_policy,
    "Joint-Wait": joint_wait_policy,
    "Cheapest-Window": cheapest_window,
    "Cleanest-Window": cleanest_window,
    "Joint-Window": joint_window,
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
    "cost_diff_vs_sigma0_pct_mean",
    "cost_diff_95ci_low_pct",
    "cost_diff_95ci_high_pct",
    "carbon_diff_vs_sigma0_pct_mean",
    "carbon_diff_95ci_low_pct",
    "carbon_diff_95ci_high_pct",
    "avg_wait_h_mean",
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
    slacks=(1, 2, 4, 8, 12, 16, 20, 24),
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


def run_ablation(
    output_csv: Path | str | None = None,
    output_png: Path | str | None = None,
    slacks=(1, 4, 12, 24),
    num_jobs: int = 60,
    window_hours: int = 120,
    job_seed: int = 42,
) -> pd.DataFrame:
    """Compare window-only and waiting variants to isolate time-shifting gains."""
    root, carbon_path, price_path = _project_paths()
    output_csv = (
        Path(output_csv) if output_csv else root / "docs" / "ablation_timeshift.csv"
    )
    output_png = (
        Path(output_png) if output_png else root / "docs" / "ablation_timeshift.png"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    objective_policies = {
        "cost": (cheapest_window, cheapest_wait_policy),
        "carbon": (cleanest_window, cleanest_wait_policy),
        "joint": (joint_window, joint_wait_policy),
    }
    rows = []
    for slack in slacks:
        jobs = JobGenerator.generate(
            num_jobs=num_jobs,
            window_hours=window_hours,
            min_slack_hours=slack,
            max_slack_hours=slack,
            seed=job_seed,
        )
        fifo = _simulate(jobs, fifo_ondemand_policy, carbon_path, price_path)
        for objective, (window_policy, wait_policy) in objective_policies.items():
            window = _simulate(jobs, window_policy, carbon_path, price_path)
            waiting = _simulate(jobs, wait_policy, carbon_path, price_path)
            window_cost = _percent_change(window["total_cost"], fifo["total_cost"])
            window_carbon = _percent_change(
                window["total_carbon_g"], fifo["total_carbon_g"]
            )
            wait_cost = _percent_change(waiting["total_cost"], fifo["total_cost"])
            wait_carbon = _percent_change(
                waiting["total_carbon_g"], fifo["total_carbon_g"]
            )
            rows.append({
                "slack": slack,
                "objective": objective,
                "window_cost_vs_fifo_pct": window_cost,
                "window_carbon_vs_fifo_pct": window_carbon,
                "wait_cost_vs_fifo_pct": wait_cost,
                "wait_carbon_vs_fifo_pct": wait_carbon,
                "time_shift_cost_effect_pp": wait_cost - window_cost,
                "time_shift_carbon_effect_pp": wait_carbon - window_carbon,
            })
    result = pd.DataFrame(rows, columns=ABLATION_COLUMNS)
    result.to_csv(output_csv, index=False, float_format="%.6f")
    _plot_ablation(result, output_png)
    return result


def run_forecast_error_sweep(
    output_csv: Path | str | None = None,
    output_png: Path | str | None = None,
    forecast_error_stds=(0, 0.1, 0.2, 0.4),
    seeds=range(20),
    num_jobs: int = 60,
    window_hours: int = 120,
    slack: int = 12,
) -> pd.DataFrame:
    """Measure joint-wait cost/carbon changes over paired seed replicates."""
    forecast_error_stds = tuple(forecast_error_stds)
    seeds = tuple(seeds)
    root, carbon_path, price_path = _project_paths()
    output_csv = (
        Path(output_csv) if output_csv else root / "docs" / "forecast_error.csv"
    )
    output_png = (
        Path(output_png) if output_png else root / "docs" / "forecast_error.png"
    )
    if 0 not in forecast_error_stds:
        raise ValueError("forecast_error_stds must include 0 as the paired baseline.")
    if not seeds:
        raise ValueError("seeds must contain at least one job seed.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    paired_samples = {
        std: {"cost": [], "carbon": [], "wait": []}
        for std in forecast_error_stds
    }
    for seed in seeds:
        jobs = JobGenerator.generate(
            num_jobs=num_jobs,
            window_hours=window_hours,
            min_slack_hours=slack,
            max_slack_hours=slack,
            seed=seed,
        )
        fifo = _simulate(jobs, fifo_ondemand_policy, carbon_path, price_path)
        per_seed = {}
        for std in forecast_error_stds:
            forecast = _simulate(
                jobs,
                joint_wait_policy,
                carbon_path,
                price_path,
                forecast_error_std=std,
                forecast_seed=seed + 10_000,
            )
            per_seed[std] = {
                "cost": _percent_change(forecast["total_cost"], fifo["total_cost"]),
                "carbon": _percent_change(
                    forecast["total_carbon_g"], fifo["total_carbon_g"]
                ),
                "wait": forecast["avg_wait_h"],
            }
        baseline = per_seed[0]
        for std in forecast_error_stds:
            paired_samples[std]["cost"].append(
                per_seed[std]["cost"] - baseline["cost"]
            )
            paired_samples[std]["carbon"].append(
                per_seed[std]["carbon"] - baseline["carbon"]
            )
            paired_samples[std]["wait"].append(per_seed[std]["wait"])

    rows = []
    for std in forecast_error_stds:
        cost_interval = _mean_t_interval(paired_samples[std]["cost"])
        carbon_interval = _mean_t_interval(paired_samples[std]["carbon"])
        rows.append({
            "forecast_error_std": std,
            "cost_diff_vs_sigma0_pct_mean": float(
                np.mean(paired_samples[std]["cost"])
            ),
            "cost_diff_95ci_low_pct": cost_interval[0],
            "cost_diff_95ci_high_pct": cost_interval[1],
            "carbon_diff_vs_sigma0_pct_mean": float(
                np.mean(paired_samples[std]["carbon"])
            ),
            "carbon_diff_95ci_low_pct": carbon_interval[0],
            "carbon_diff_95ci_high_pct": carbon_interval[1],
            "avg_wait_h_mean": float(np.mean(paired_samples[std]["wait"])),
        })
    result = pd.DataFrame(rows, columns=FORECAST_ERROR_COLUMNS)
    result.to_csv(output_csv, index=False, float_format="%.6f")
    _plot_forecast_error(result, output_png)
    return result


def _percent_change(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if value == 0 else float("inf")
    return 100 * (value - baseline) / baseline


def _mean_t_interval(values) -> tuple[float, float]:
    """Return a two-sided 95% Student t interval for the sample mean."""
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return float("nan"), float("nan")
    critical_value = _student_t_critical_95(len(values) - 1)
    half_width = critical_value * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean - half_width, mean + half_width


def _student_t_critical_95(degrees_of_freedom: int) -> float:
    """Return the 97.5th percentile for a two-sided 95% t interval."""
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093024054,
        20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
        30: 2.042,
    }
    return table.get(degrees_of_freedom, 1.960)


ABLATION_COLUMNS = [
    "slack",
    "objective",
    "window_cost_vs_fifo_pct",
    "window_carbon_vs_fifo_pct",
    "wait_cost_vs_fifo_pct",
    "wait_carbon_vs_fifo_pct",
    "time_shift_cost_effect_pp",
    "time_shift_carbon_effect_pp",
]


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
        ax.legend(
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
        )
    fig.suptitle("Policy impact of scheduling slack (synthetic traces)")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_forecast_error(result: pd.DataFrame, output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        result["forecast_error_std"],
        result["cost_diff_vs_sigma0_pct_mean"],
        yerr=[
            result["cost_diff_vs_sigma0_pct_mean"]
            - result["cost_diff_95ci_low_pct"],
            result["cost_diff_95ci_high_pct"]
            - result["cost_diff_vs_sigma0_pct_mean"],
        ],
        marker="o",
        capsize=4,
        label="Paired cost difference vs sigma=0 (95% t CI)",
    )
    ax.errorbar(
        result["forecast_error_std"],
        result["carbon_diff_vs_sigma0_pct_mean"],
        yerr=[
            result["carbon_diff_vs_sigma0_pct_mean"]
            - result["carbon_diff_95ci_low_pct"],
            result["carbon_diff_95ci_high_pct"]
            - result["carbon_diff_vs_sigma0_pct_mean"],
        ],
        marker="o",
        capsize=4,
        label="Paired carbon difference vs sigma=0 (95% t CI)",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Forecast error standard deviation")
    ax.set_ylabel("Difference in vs-FIFO percentage points")
    ax.set_title("Joint-Wait forecast-error sensitivity (12 h slack)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_ablation(result: pd.DataFrame, output_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"cost": "#4c78a8", "carbon": "#54a24b", "joint": "#e45756"}
    for ax, value_column, title in (
        (axes[0], "time_shift_cost_effect_pp", "Time-shifting effect on cost"),
        (axes[1], "time_shift_carbon_effect_pp", "Time-shifting effect on carbon"),
    ):
        for objective, rows in result.groupby("objective", sort=False):
            rows = rows.sort_values("slack")
            ax.plot(
                rows["slack"],
                rows[value_column],
                marker="o",
                label=objective.title(),
                color=colors[objective],
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Fixed slack (hours)")
        ax.set_ylabel("Wait minus Window (percentage points)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Pure time-shifting effect (synthetic traces)")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    slack_results = run_slack_sweep()
    ablation_results = run_ablation()
    forecast_results = run_forecast_error_sweep()
    print("\nSlack sweep CSV head:")
    print(slack_results.head().to_csv(index=False), end="")
    print("\nForecast-error sweep CSV head:")
    print(forecast_results.head().to_csv(index=False), end="")
    print("\nTime-shifting ablation CSV head:")
    print(ablation_results.head().to_csv(index=False), end="")
