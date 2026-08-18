"""
Compare the four scheduling policies across key metrics.
Runs simulator with each policy using the same job set and generates
comparison charts and statistics.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from jobs.job import JobGenerator
from simulator.simulator import Simulator
from scheduler.policies import (
    fifo_ondemand_policy,
    cheapest_first_policy,
    cleanest_first_policy,
    joint_policy,
)


def run_comparison():
    """Run all four policies on the same job set and compare results."""
    
    # Set seed for reproducibility
    random.seed(42)
    
    # Generate one fixed job list to use across all four runs
    # Use tighter deadlines to create scheduling challenges
    print("Generating fixed job set for fair comparison...")
    jobs = JobGenerator.generate(
        num_jobs=60,
        window_hours=168,
        min_slack_hours=0.5,  # Tighter deadlines
        max_slack_hours=3.0   # Less flexibility
    )
    
    # Define the four policies
    policies = {
        "FIFO OnDemand": fifo_ondemand_policy,
        "Cheapest First": cheapest_first_policy,
        "Cleanest First": cleanest_first_policy,
        "Joint Policy": joint_policy,
    }
    
    # Run simulator for each policy
    results = {}
    print("\nRunning simulator with each policy...\n")
    
    for policy_name, policy_func in policies.items():
        print(f"  Running {policy_name}...", end=" ", flush=True)
        
        # Initialize simulator (fresh instance for each run)
        sim = Simulator(
            carbon_csv_path="data/carbon_traces/sample_carbon.csv",
            price_csv_path="data/price_traces/sample_price.csv",
        )
        
        # Run simulation
        results_df = sim.run(jobs, policy_func)
        
        # Collect metrics
        total_cost = results_df["cost_incurred"].sum()
        total_carbon = results_df["carbon_emitted_g"].sum()
        deadline_rate = 100 * results_df["met_deadline"].sum() / len(results_df)
        avg_completion_time = results_df[
            results_df["completion_hour"].notna()
        ]["completion_hour"].mean()
        
        results[policy_name] = {
            "results_df": results_df,
            "total_cost": total_cost,
            "total_carbon": total_carbon,
            "deadline_success_rate": deadline_rate,
            "avg_completion_time": avg_completion_time,
        }
        
        print(f"✓ (Cost: ${total_cost:.2f}, Carbon: {total_carbon:.0f}g, Deadline: {deadline_rate:.1f}%)")
    
    # Create comparison table
    print("\n" + "="*100)
    print("POLICY COMPARISON TABLE")
    print("="*100)
    
    comparison_data = []
    for policy_name, metrics in results.items():
        comparison_data.append({
            "Policy": policy_name,
            "Total Cost ($)": round(metrics["total_cost"], 2),
            "Total Carbon (g CO2)": int(metrics["total_carbon"]),
            "Deadline Success Rate (%)": round(metrics["deadline_success_rate"], 1),
            "Avg Completion Time (hrs)": round(metrics["avg_completion_time"], 1),
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Calculate improvements
    print("\n" + "="*100)
    print("IMPROVEMENT OVER FIFO BASELINE")
    print("="*100)
    
    fifo_cost = results["FIFO OnDemand"]["total_cost"]
    fifo_carbon = results["FIFO OnDemand"]["total_carbon"]
    
    improvements = []
    for policy_name, metrics in results.items():
        if policy_name != "FIFO OnDemand":
            cost_improvement = 100 * (fifo_cost - metrics["total_cost"]) / fifo_cost
            carbon_improvement = 100 * (fifo_carbon - metrics["total_carbon"]) / fifo_carbon
            
            improvements.append({
                "Policy": policy_name,
                "Cost Savings (%)": round(cost_improvement, 1),
                "Carbon Reduction (%)": round(carbon_improvement, 1),
            })
    
    improvements_df = pd.DataFrame(improvements)
    print(improvements_df.to_string(index=False))
    
    # Save comparison table to CSV
    csv_path = "docs/policy_comparison.csv"
    os.makedirs("docs", exist_ok=True)
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved comparison table to {csv_path}")
    
    # Create visualization
    print("\nGenerating comparison charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Scheduling Policy Comparison", fontsize=16, fontweight="bold")
    
    policy_names = list(results.keys())
    
    # Chart 1: Total Cost
    costs = [results[p]["total_cost"] for p in policy_names]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ax = axes[0, 0]
    bars = ax.bar(policy_names, costs, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Total Cost ($)", fontweight="bold")
    ax.set_title("Total Cost by Policy", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        ax.text(bar.get_x() + bar.get_width()/2, cost + 5, f"${cost:.2f}", 
                ha="center", va="bottom", fontweight="bold")
    
    # Chart 2: Total Carbon
    carbons = [results[p]["total_carbon"] for p in policy_names]
    ax = axes[0, 1]
    bars = ax.bar(policy_names, carbons, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Total Carbon Emissions (g CO2)", fontweight="bold")
    ax.set_title("Total Carbon Emissions by Policy", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, (bar, carbon) in enumerate(zip(bars, carbons)):
        ax.text(bar.get_x() + bar.get_width()/2, carbon + 500, f"{carbon:.0f}g", 
                ha="center", va="bottom", fontweight="bold")
    
    # Chart 3: Deadline Success Rate
    deadline_rates = [results[p]["deadline_success_rate"] for p in policy_names]
    ax = axes[1, 0]
    bars = ax.bar(policy_names, deadline_rates, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Deadline Success Rate (%)", fontweight="bold")
    ax.set_title("Deadline Success Rate by Policy", fontweight="bold")
    ax.set_ylim([max(0, min(deadline_rates) - 5), 105])
    ax.grid(axis="y", alpha=0.3)
    for i, (bar, rate) in enumerate(zip(bars, deadline_rates)):
        ax.text(bar.get_x() + bar.get_width()/2, rate + 0.1, f"{rate:.1f}%", 
                ha="center", va="bottom", fontweight="bold")
    
    # Chart 4: Side-by-side Cost vs Carbon comparison
    ax = axes[1, 1]
    x = np.arange(len(policy_names))
    width = 0.35
    
    # Normalize cost and carbon to 0-100 scale for comparison
    costs_normalized = 100 * np.array(costs) / max(costs)
    carbons_normalized = 100 * np.array(carbons) / max(carbons)
    
    bars1 = ax.bar(x - width/2, costs_normalized, width, label="Cost (normalized)", 
                   color="#1f77b4", alpha=0.7, edgecolor="black", linewidth=1.5)
    bars2 = ax.bar(x + width/2, carbons_normalized, width, label="Carbon (normalized)", 
                   color="#2ca02c", alpha=0.7, edgecolor="black", linewidth=1.5)
    
    ax.set_ylabel("Normalized Score (0-100)", fontweight="bold")
    ax.set_title("Cost vs Carbon Comparison (Normalized)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(policy_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    png_path = "docs/policy_comparison.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved comparison chart to {png_path}")
    plt.close()
    
    # Print summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"✓ Ran all 4 policies on a fixed set of {len(jobs)} jobs")
    print(f"✓ Joint Policy cost: ${results['Joint Policy']['total_cost']:.2f} " +
          f"({100*(fifo_cost-results['Joint Policy']['total_cost'])/fifo_cost:.1f}% savings vs FIFO)")
    print(f"✓ Joint Policy carbon: {results['Joint Policy']['total_carbon']:.0f}g CO2 " +
          f"({100*(fifo_carbon-results['Joint Policy']['total_carbon'])/fifo_carbon:.1f}% reduction vs FIFO)")
    print(f"✓ Deadline success rates: " + ", ".join(f"{p}: {r['deadline_success_rate']:.1f}%" for p, r in results.items()))
    print(f"\n✓ Results saved to:")
    print(f"  - {csv_path}")
    print(f"  - {png_path}")


if __name__ == "__main__":
    run_comparison()
