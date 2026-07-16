# GreenSched

A research project skeleton for carbon-aware scheduling experiments.

## Project layout

- src/jobs: job implementations for OpenMP and CUDA/embedding work
- src/models: embedding model definitions
- src/scheduler: scheduling policies
- src/simulator: discrete-event simulation core
- src/metrics: carbon, cost, and SLA calculations
- data/carbon_traces: carbon trace CSVs (ignored by git)
- data/price_traces: electricity price trace CSVs (ignored by git)
- notebooks: analysis and plotting notebooks
- docs: papers and diagrams

## Data setup

Raw trace CSV files are intentionally excluded from version control to avoid bloating the repository. Place downloaded CSV files in the appropriate data directory before running experiments, or add a fetch script to populate them automatically.
