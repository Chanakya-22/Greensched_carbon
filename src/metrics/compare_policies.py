"""Compare immediate and forecast-aware policies on synthetic trace scenarios."""
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
from jobs.job import JobGenerator
from simulator.simulator import Simulator
from scheduler.policies import (
    fifo_ondemand_policy,
    cheapest_first_policy,
    cleanest_first_policy,
    joint_policy,
    cheapest_wait_policy,
    cleanest_wait_policy,
    joint_wait_policy,
)


def run_comparison():
    """Run two seeded slack scenarios and write their tables and charts."""
    policies = {
        "FIFO OnDemand": fifo_ondemand_policy,
        "Cheapest First": cheapest_first_policy,
        "Cleanest First": cleanest_first_policy,
        "Joint Policy": joint_policy,
        "Cheapest-Wait": cheapest_wait_policy,
        "Cleanest-Wait": cleanest_wait_policy,
        "Joint-Wait": joint_wait_policy,
    }
    scenarios = {
        "Tight slack": (0.5, 3.0, ""),
        "Relaxed slack": (2.0, 12.0, "_relaxed"),
    }
    project_root = Path(__file__).resolve().parents[2]
    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)
    carbon_path = project_root / "data/carbon_traces/sample_carbon.csv"
    price_path = project_root / "data/price_traces/sample_price.csv"
    results_by_scenario = {}

    print(
        "Policy comparison: synthetic traces; arrivals window=120 hours; "
        "job seed=42; perfect forecast (std=0); forecast horizon=24 hours."
    )
    print("Cost and carbon percentages are signed vs FIFO; negative = better.")
    print("Average wait is calculated across scheduled jobs; deadline success uses all jobs.")

    for scenario_name, (min_slack, max_slack, file_suffix) in scenarios.items():
        jobs = JobGenerator.generate(
            num_jobs=60,
            window_hours=120,
            min_slack_hours=min_slack,
            max_slack_hours=max_slack,
            seed=42,
        )
        scenario_results = {}
        for policy_name, policy_func in policies.items():
            simulator = Simulator(
                carbon_csv_path=str(carbon_path),
                price_csv_path=str(price_path),
                forecast_error_std=0.0,
                forecast_horizon_hours=24,
                forecast_seed=42,
            )
            results_df = simulator.run(jobs, policy_func)
            scheduled = results_df[results_df["region_used"].notna()]
            scenario_results[policy_name] = {
                "total_cost": results_df["cost_incurred"].sum(),
                "total_carbon": results_df["carbon_emitted_g"].sum(),
                "average_wait": scheduled["wait_hours"].mean(),
                "deadline_success": (
                    100 * results_df["met_deadline"].sum() / len(results_df)
                ),
            }

        fifo = scenario_results["FIFO OnDemand"]
        rows = []
        for policy_name, metrics in scenario_results.items():
            rows.append({
                "Policy": policy_name,
                "Total Cost (USD)": metrics["total_cost"],
                "Total Carbon (gCO2eq)": metrics["total_carbon"],
                "Average Wait (hours)": metrics["average_wait"],
                "Deadline Success (%)": metrics["deadline_success"],
                "Cost vs FIFO (%)": (
                    100 * (metrics["total_cost"] - fifo["total_cost"])
                    / fifo["total_cost"]
                ),
                "Carbon vs FIFO (%)": (
                    100 * (metrics["total_carbon"] - fifo["total_carbon"])
                    / fifo["total_carbon"]
                ),
            })

        comparison_df = pd.DataFrame(rows)
        results_by_scenario[scenario_name] = comparison_df
        formatters = {
            "Total Cost (USD)": lambda value: f"{value:.2f}",
            "Total Carbon (gCO2eq)": lambda value: f"{value:.0f}",
            "Average Wait (hours)": lambda value: f"{value:.2f}",
            "Deadline Success (%)": lambda value: f"{value:.1f}%",
            "Cost vs FIFO (%)": lambda value: f"{value:+.1f}%",
            "Carbon vs FIFO (%)": lambda value: f"{value:+.1f}%",
        }
        print(f"\n{scenario_name.upper()} ({min_slack:g}-{max_slack:g} h slack)")
        print(comparison_df.to_string(index=False, formatters=formatters))

        csv_path = docs_dir / f"policy_comparison{file_suffix}.csv"
        comparison_df.to_csv(csv_path, index=False, float_format="%.6f")
        png_path = docs_dir / f"policy_comparison{file_suffix}.png"
        _save_scenario_chart(
            scenario_name,
            scenario_results,
            str(png_path),
        )
        print(f"Saved: {csv_path.relative_to(project_root)}")
        print(f"Saved: {png_path.relative_to(project_root)}")

    return results_by_scenario


def _save_scenario_chart(scenario_name, results, output_path):
    """Write a readable four-metric chart with horizontal policy labels."""
    policy_names = list(results)
    colors = [
        "#4c78a8", "#f58518", "#54a24b", "#e45756",
        "#72b7b2", "#b279a2", "#ff9da6",
    ]
    metrics = [
        ("total_cost", "Total Cost (USD)"),
        ("total_carbon", "Total Carbon (gCO2eq)"),
        ("average_wait", "Average Wait (hours)"),
        ("deadline_success", "Deadline Success (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{scenario_name}: seven scheduling policies",
        fontsize=16,
        fontweight="bold",
    )
    for ax, (metric, title) in zip(axes.flat, metrics):
        values = [results[name][metric] for name in policy_names]
        ax.barh(policy_names, values, color=colors, edgecolor="black", alpha=0.8)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_comparison()
