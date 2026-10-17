"""Run seeded spot-risk and on-demand price experiments.

Experiments use generated synthetic traces rather than measured grid-carbon or
electricity-price data.
"""
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jobs.job import JobGenerator
from metrics.slack_sweep import _mean_t_interval
from scheduler.policies import (
    market_cheapest,
    market_joint,
    ondemand_only_fifo,
    spot_fifo,
    spot_only_joint_wait,
)
from simulator.simulator import Simulator


POLICIES = {
    "ondemand_only_fifo": ondemand_only_fifo,
    "spot_fifo": spot_fifo,
    "spot_only_joint_wait": spot_only_joint_wait,
    "market_joint": market_joint,
    "market_cheapest": market_cheapest,
}
METRICS = (
    "cost",
    "carbon_g",
    "deadline_success_pct",
    "mean_interruptions",
    "wasted_cost",
    "wasted_carbon_g",
)
SUMMARY_COLUMNS = [
    "parameter",
    "policy",
    "n",
    "cost_mean",
    "cost_ci_low",
    "cost_ci_high",
    "carbon_g_mean",
    "carbon_g_ci_low",
    "carbon_g_ci_high",
    "deadline_success_pct_mean",
    "deadline_success_pct_ci_low",
    "deadline_success_pct_ci_high",
    "mean_interruptions_mean",
    "mean_interruptions_ci_low",
    "mean_interruptions_ci_high",
    "wasted_cost_mean",
    "wasted_cost_ci_low",
    "wasted_cost_ci_high",
    "wasted_carbon_g_mean",
    "wasted_carbon_g_ci_low",
    "wasted_carbon_g_ci_high",
    "cost_vs_ondemand_fifo_pct_mean",
    "cost_vs_ondemand_fifo_pct_ci_low",
    "cost_vs_ondemand_fifo_pct_ci_high",
    "carbon_vs_ondemand_fifo_pct_mean",
    "carbon_vs_ondemand_fifo_pct_ci_low",
    "carbon_vs_ondemand_fifo_pct_ci_high",
    "cost_vs_spot_fifo_pct_mean",
    "cost_vs_spot_fifo_pct_ci_low",
    "cost_vs_spot_fifo_pct_ci_high",
    "carbon_vs_spot_fifo_pct_mean",
    "carbon_vs_spot_fifo_pct_ci_low",
    "carbon_vs_spot_fifo_pct_ci_high",
]


def _project_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    return (
        root,
        root / "data" / "carbon_traces" / "sample_carbon.csv",
        root / "data" / "price_traces" / "sample_price.csv",
    )


def _run_one(
    jobs,
    policy,
    carbon_path: Path,
    price_path: Path,
    hazard: float,
    multiplier: float,
    seed: int,
) -> dict:
    hazards = {
        "region-cheap-dirty": hazard,
        "region-balanced": hazard / 2,
        "region-green-premium": hazard / 4,
    }
    simulator = Simulator(
        carbon_csv_path=str(carbon_path),
        price_csv_path=str(price_path),
        include_on_demand=True,
        on_demand_multiplier=multiplier,
        interruption_hazard=hazards,
        interruption_seed=seed,
        forecast_horizon_hours=24,
        forecast_error_std=0.0,
        forecast_seed=seed,
    )
    results = simulator.run(jobs, policy)
    return {
        "cost": float(results["cost_incurred"].sum()),
        "carbon_g": float(results["carbon_emitted_g"].sum()),
        "deadline_success_pct": float(100 * results["met_deadline"].mean()),
        "mean_interruptions": float(results["interruptions"].mean()),
        "wasted_cost": float(results["wasted_cost"].sum()),
        "wasted_carbon_g": float(results["wasted_carbon_g"].sum()),
    }


def _percent_change(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if value == 0 else float("inf")
    return 100 * (value - baseline) / baseline


def _aggregate_runs(
    parameter_values,
    seeds,
    run_metrics: dict,
) -> pd.DataFrame:
    rows = []
    for parameter in parameter_values:
        ondemand_baseline_by_seed = {
            seed: run_metrics[(parameter, seed, "ondemand_only_fifo")]
            for seed in seeds
        }
        spot_baseline_by_seed = {
            seed: run_metrics[(parameter, seed, "spot_fifo")]
            for seed in seeds
        }
        for policy_name in POLICIES:
            samples = {
                metric: [
                    run_metrics[(parameter, seed, policy_name)][metric]
                    for seed in seeds
                ]
                for metric in METRICS
            }
            row = {
                "parameter": parameter,
                "policy": policy_name,
                "n": len(seeds),
            }
            for metric, values in samples.items():
                mean = float(np.mean(values))
                low, high = _mean_t_interval(values)
                if metric == "deadline_success_pct":
                    low = max(0.0, low)
                    high = min(100.0, high)
                elif metric == "mean_interruptions":
                    low = max(0.0, low)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            for baseline_name, baseline_by_seed in (
                ("ondemand_fifo", ondemand_baseline_by_seed),
                ("spot_fifo", spot_baseline_by_seed),
            ):
                for metric in ("cost", "carbon_g"):
                    values = [
                        _percent_change(
                            run_metrics[(parameter, seed, policy_name)][metric],
                            baseline_by_seed[seed][metric],
                        )
                        for seed in seeds
                    ]
                    metric_stem = "carbon" if metric == "carbon_g" else metric
                    metric_name = f"{metric_stem}_vs_{baseline_name}_pct"
                    mean = float(np.mean(values))
                    low, high = _mean_t_interval(values)
                    row[f"{metric_name}_mean"] = mean
                    row[f"{metric_name}_ci_low"] = low
                    row[f"{metric_name}_ci_high"] = high
            rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _plot_sweep(
    result: pd.DataFrame,
    output_png: Path,
    parameter_label: str,
    title: str,
) -> None:
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = list(axes.flat)
    plot_metrics = (
        (
            "cost_vs_ondemand_fifo_pct",
            "Cost vs on-demand FIFO (%)",
            "Change (%)",
        ),
        (
            "carbon_vs_ondemand_fifo_pct",
            "Carbon vs on-demand FIFO (%)",
            "Change (%)",
        ),
        ("cost_vs_spot_fifo_pct", "Cost vs spot FIFO (%)", "Change (%)"),
        ("carbon_vs_spot_fifo_pct", "Carbon vs spot FIFO (%)", "Change (%)"),
        ("deadline_success_pct", "Deadline success (%)", "Success (%)"),
    )
    for ax, (metric, heading, ylabel) in zip(axes, plot_metrics):
        for index, (policy, rows) in enumerate(
            result.groupby("policy", sort=False)
        ):
            rows = rows.sort_values("parameter")
            mean_column = f"{metric}_mean"
            low_column = f"{metric}_ci_low"
            high_column = f"{metric}_ci_high"
            means = rows[mean_column].to_numpy()
            yerr = np.vstack((
                means - rows[low_column].to_numpy(),
                rows[high_column].to_numpy() - means,
            ))
            ax.errorbar(
                rows["parameter"],
                means,
                yerr=yerr,
                marker="o",
                capsize=3,
                label=policy,
                color=colors[index % len(colors)],
            )
        if metric != "deadline_success_pct":
            ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(heading)
        ax.set_xlabel(parameter_label)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes[-1].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.99, 0.5),
        borderaxespad=0,
    )
    fig.suptitle(f"{title} (synthetic traces; 95% t confidence intervals)")
    fig.tight_layout(rect=(0, 0, 0.87, 0.94))
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _run_sweep(
    parameter_name: str,
    parameter_values,
    fixed_hazard: float | None,
    fixed_multiplier: float | None,
    seeds,
    num_jobs: int,
    window_hours: int,
    output_csv: Path,
    output_png: Path,
    parameter_label: str,
    title: str,
) -> pd.DataFrame:
    root, carbon_path, price_path = _project_paths()
    del root
    seeds = tuple(seeds)
    parameter_values = tuple(parameter_values)
    if not seeds:
        raise ValueError("At least one seed is required.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    runs = {}
    for parameter in parameter_values:
        hazard = parameter if parameter_name == "hazard" else fixed_hazard
        multiplier = (
            parameter if parameter_name == "multiplier" else fixed_multiplier
        )
        for seed in seeds:
            jobs = JobGenerator.generate(
                num_jobs=num_jobs,
                window_hours=window_hours,
                min_slack_hours=2,
                max_slack_hours=12,
                seed=seed,
            )
            for policy_name, policy in POLICIES.items():
                runs[(parameter, seed, policy_name)] = _run_one(
                    jobs,
                    policy,
                    carbon_path,
                    price_path,
                    hazard,
                    multiplier,
                    seed,
                )

    result = _aggregate_runs(parameter_values, seeds, runs)
    result.to_csv(output_csv, index=False, float_format="%.6f")
    _plot_sweep(result, output_png, parameter_label, title)
    return result


def run_spot_hazard_sweep(
    output_csv: Path | str | None = None,
    output_png: Path | str | None = None,
    hazards=(0, 0.02, 0.05, 0.1, 0.2),
    seeds=range(10),
    num_jobs: int = 60,
    window_hours: int = 120,
) -> pd.DataFrame:
    """Measure market policies across spot reclaim hazards."""
    root, _, _ = _project_paths()
    return _run_sweep(
        "hazard",
        hazards,
        fixed_hazard=None,
        fixed_multiplier=2.5,
        seeds=seeds,
        num_jobs=num_jobs,
        window_hours=window_hours,
        output_csv=(
            Path(output_csv)
            if output_csv
            else root / "docs" / "spot_hazard_sweep.csv"
        ),
        output_png=(
            Path(output_png)
            if output_png
            else root / "docs" / "spot_hazard_sweep.png"
        ),
        parameter_label="Cheap-dirty region spot reclaim hazard",
        title="Spot interruption hazard sweep",
    )


def run_spot_multiplier_sweep(
    output_csv: Path | str | None = None,
    output_png: Path | str | None = None,
    multipliers=(1.5, 2.5, 4.0),
    hazard: float = 0.05,
    seeds=range(10),
    num_jobs: int = 60,
    window_hours: int = 120,
) -> pd.DataFrame:
    """Measure market policies across on-demand price multipliers."""
    root, _, _ = _project_paths()
    return _run_sweep(
        "multiplier",
        multipliers,
        fixed_hazard=hazard,
        fixed_multiplier=None,
        seeds=seeds,
        num_jobs=num_jobs,
        window_hours=window_hours,
        output_csv=(
            Path(output_csv)
            if output_csv
            else root / "docs" / "spot_multiplier_sweep.csv"
        ),
        output_png=(
            Path(output_png)
            if output_png
            else root / "docs" / "spot_multiplier_sweep.png"
        ),
        parameter_label="On-demand price multiplier",
        title="On-demand multiplier sweep (hazard=0.05)",
    )


def run_spot_experiments():
    """Run both requested full-size sweeps and return their dataframes."""
    hazard = run_spot_hazard_sweep()
    multiplier = run_spot_multiplier_sweep()
    return hazard, multiplier


if __name__ == "__main__":
    hazard_results, multiplier_results = run_spot_experiments()
    print("Hazard sweep CSV head:")
    print(hazard_results.head().to_csv(index=False), end="")
    print("\nMultiplier sweep CSV head:")
    print(multiplier_results.head().to_csv(index=False), end="")
