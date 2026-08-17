"""
Synthetic data generator for carbon intensity and spot prices.
Generates realistic multi-region traces with daily patterns and random variations.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict


def generate_region_traces(regions: List[Dict], hours: int = 168) -> tuple:
    """
    Generate hourly carbon intensity and spot price traces for multiple regions.
    
    Args:
        regions: List of dicts like:
            {
                "name": "region-A",
                "carbon_base": 250,  # gCO2/kWh baseline
                "carbon_amplitude": 150,  # variation amplitude
                "price_cpu_base": 0.05,  # $/hr baseline for CPU
                "price_gpu_base": 0.6,  # $/hr baseline for GPU
            }
        hours: Number of hourly measurements to generate (default 1 week = 168)
        
    Returns:
        Tuple of (combined_df, carbon_df, price_df)
    """
    all_rows = []
    
    # Start time
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    
    for region in regions:
        region_name = region["name"]
        carbon_base = region["carbon_base"]
        carbon_amplitude = region["carbon_amplitude"]
        price_cpu_base = region["price_cpu_base"]
        price_gpu_base = region["price_gpu_base"]
        
        # Generate hourly timestamps
        for hour in range(hours):
            timestamp = start_time + timedelta(hours=hour)
            hour_of_day = timestamp.hour
            
            # Carbon intensity: daily cosine pattern (low 3-5am, high 6-9pm)
            # Shifted so minimum is around hour 4 (4am)
            carbon_cycle = -np.cos((hour_of_day - 4) * 2 * np.pi / 24)
            carbon_noise = np.random.uniform(-0.08, 0.08)  # +/- 8% noise
            carbon_intensity = carbon_base + carbon_amplitude * carbon_cycle
            carbon_intensity *= (1 + carbon_noise)
            carbon_intensity = max(0, carbon_intensity)  # Keep non-negative
            
            # CPU price: smaller daily variation + occasional spikes
            price_cpu_cycle = -np.cos((hour_of_day - 4) * 2 * np.pi / 24) * 0.15
            price_cpu = price_cpu_base + price_cpu_cycle
            
            # 5% chance of price spike (demand surge): 1.5x to 2.5x multiplier
            if np.random.random() < 0.05:
                spike_multiplier = np.random.uniform(1.5, 2.5)
                price_cpu *= spike_multiplier
            
            price_cpu = max(0.01, price_cpu)  # Keep positive
            
            # GPU price: similar pattern but larger base
            price_gpu_cycle = -np.cos((hour_of_day - 4) * 2 * np.pi / 24) * 1.8
            price_gpu = price_gpu_base + price_gpu_cycle
            
            if np.random.random() < 0.05:
                spike_multiplier = np.random.uniform(1.5, 2.5)
                price_gpu *= spike_multiplier
            
            price_gpu = max(0.1, price_gpu)  # Keep positive
            
            all_rows.append({
                "region": region_name,
                "timestamp": timestamp,
                "hour_of_day": hour_of_day,
                "carbon_intensity_gco2_kwh": carbon_intensity,
                "price_cpu_usd_hr": price_cpu,
                "price_gpu_usd_hr": price_gpu,
            })
    
    # Create combined DataFrame
    df = pd.DataFrame(all_rows)
    
    # Split into separate dataframes for carbon and price
    carbon_cols = ["region", "timestamp", "carbon_intensity_gco2_kwh"]
    carbon_df = df[carbon_cols].copy()
    
    price_cols = ["region", "timestamp", "price_cpu_usd_hr", "price_gpu_usd_hr"]
    price_df = df[price_cols].copy()
    
    return df, carbon_df, price_df


def save_traces(carbon_df: pd.DataFrame, price_df: pd.DataFrame) -> None:
    """Save carbon and price traces to CSV files."""
    # Ensure directories exist
    os.makedirs("data/carbon_traces", exist_ok=True)
    os.makedirs("data/price_traces", exist_ok=True)
    
    # Save files
    carbon_path = "data/carbon_traces/sample_carbon.csv"
    price_path = "data/price_traces/sample_price.csv"
    
    carbon_df.to_csv(carbon_path, index=False)
    price_df.to_csv(price_path, index=False)
    
    print(f"✓ Saved carbon traces to {carbon_path}")
    print(f"✓ Saved price traces to {price_path}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define three regions with clearly different profiles
    regions = [
        {
            "name": "region-clean-night",
            "carbon_base": 200,      # Cleaner baseline
            "carbon_amplitude": 100,  # Moderate variation
            "price_cpu_base": 0.03,
            "price_gpu_base": 0.4,
        },
        {
            "name": "region-volatile",
            "carbon_base": 350,      # Higher baseline
            "carbon_amplitude": 200,  # High variation (very dirty evenings)
            "price_cpu_base": 0.08,
            "price_gpu_base": 0.9,
        },
        {
            "name": "region-midrange",
            "carbon_base": 280,      # Middle baseline
            "carbon_amplitude": 120,  # Medium variation
            "price_cpu_base": 0.05,
            "price_gpu_base": 0.6,
        },
    ]
    
    # Generate traces for 1 week (168 hours)
    print("Generating synthetic carbon and price traces...")
    combined_df, carbon_df, price_df = generate_region_traces(regions, hours=168)
    
    # Save to CSVs
    save_traces(carbon_df, price_df)
    
    # Print summary
    print("\n" + "="*60)
    print("CARBON INTENSITY SUMMARY (gCO2/kWh)")
    print("="*60)
    carbon_summary = combined_df.groupby("region")["carbon_intensity_gco2_kwh"].agg(
        ["min", "mean", "max", "std"]
    ).round(2)
    print(carbon_summary)
    
    print("\n" + "="*60)
    print("CPU PRICE SUMMARY ($/hr)")
    print("="*60)
    cpu_summary = combined_df.groupby("region")["price_cpu_usd_hr"].agg(
        ["min", "mean", "max", "std"]
    ).round(4)
    print(cpu_summary)
    
    print("\n" + "="*60)
    print("GPU PRICE SUMMARY ($/hr)")
    print("="*60)
    gpu_summary = combined_df.groupby("region")["price_gpu_usd_hr"].agg(
        ["min", "mean", "max", "std"]
    ).round(2)
    print(gpu_summary)
    
    # Visual sanity check: show daily pattern for one region
    print("\n" + "="*60)
    print("SAMPLE: Carbon intensity pattern for region-clean-night (first 24 hours)")
    print("="*60)
    sample = combined_df[
        (combined_df["region"] == "region-clean-night") & 
        (combined_df["timestamp"] < datetime(2024, 1, 2))
    ].reset_index(drop=True)
    print(sample[["timestamp", "hour_of_day", "carbon_intensity_gco2_kwh"]].to_string())
