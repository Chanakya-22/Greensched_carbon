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

## Quick start

From the repository root, create and activate a virtual environment, install the
core dependencies, generate the sample traces, and run the policy comparison:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/data/synthetic_data.py
python src/metrics/compare_policies.py
```

The trace CSVs are generated locally by `src/data/synthetic_data.py` and are
not committed to the repository. They are synthetic sample data, not real
observed grid-carbon or electricity-price data; no real trace data is included
in this project skeleton.
