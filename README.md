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

## AI inference model selection

Optional image/text embedding dependencies are listed in
`requirements-models.txt`. Install them with
`pip install -r requirements-models.txt`. The registry uses published
pretrained OpenCLIP weights; the first run downloads model weights that are
not already cached. The simulator's grid-carbon and electricity-price traces
remain synthetic and are separate from these model weights and the public
benchmark images.

Run the full comparison from the repository root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
python src/models/benchmark.py
```

The default run evaluates all four registered models on the full dataset.
Use `--limit N`, `--models name1,name2`, and `--out-dir path` for smaller
or targeted comparisons.

### Dataset and metrics

The comparison uses the Flickr30k 1K test image-text retrieval dataset
(`nlphuji/flickr_1k_test_image_text_retrieval`): 1,000 images, each paired
with five captions, evaluated in the dataset's fixed order. The loader uses
the dataset's `image` and `caption` fields; when the current `datasets`
package no longer accepts its loading script, it reads the dataset's published
CSV and image archive directly. It falls back to CIFAR-100 classification
only if neither retrieval loading path is available, and reports the dataset
and task actually used in the benchmark output.

Retrieval metrics are image-to-text recall@1/5/10 (a hit means any matching
caption is retrieved), text-to-image recall@1/5/10, and their mean, with
per-query bootstrap 95% confidence intervals. For the CIFAR-100 fallback,
the runner reports top-1 and top-5 classification accuracy with bootstrap
intervals instead. Performance measurements include median and IQR image
encoding latency, images per second at batch sizes 1, 16, and 64, peak GPU
memory, parameter count, embedding dimension, and offline energy per 1,000
images.

### Hardware and inference

The full run was performed on Windows 11 with an Intel CPU and CPU-only
PyTorch 2.13.0; no CUDA-capable GPU or GPU memory measurement was available.
The CSV and profile JSON record hardware, PyTorch/OpenCLIP versions, and
timestamp. Inference uses fp16 autocast on CUDA and fp32 on CPU. On this
Windows host, CodeCarbon measured wall-clock duration and estimated CPU, RAM,
and facility energy; the CSV and profile identify measured and estimated
components separately.

### Registered models

- **ViT-B-32 OpenAI** (`vit_b_32_openai`): OpenAI-pretrained CLIP ViT-B/32
  using the QuickGELU architecture variant.
- **ViT-B-32 LAION-2B** (`vit_b_32_laion2b`): LAION-2B-pretrained CLIP
  ViT-B/32.
- **ViT-B-16-SigLIP WebLI** (`vit_b_16_siglip_webli`): SigLIP ViT-B/16
  pretrained on WebLI.
- **ConvNeXt-Base-W LAION-2B** (`convnext_base_w_laion2b`): LAION-2B-pretrained
  ConvNeXt-Base-W image/text encoder.

### Selection rule and results

Benchmark device: CPU. Energy method: estimated_from_runtime (CodeCarbon
PUE=1.0). The winner is the model with the highest mean recall@5. Models within
one percentage point of the best score are treated as tied; among tied models,
the one with the lowest energy per 1,000 images wins. The next eligible model
is the runner-up. Results below come from the full 1,000-image retrieval run;
the CSV contains all metrics, confidence-interval bounds, dataset identity,
and hardware details.

| Model | Mean recall@5 (95% CI) | Images/s (batch 64) | Energy (kWh/1,000 images) | Parameters |
| --- | ---: | ---: | ---: | ---: |
| ViT-B-32 OpenAI | 0.8864 (0.8772–0.8951) | 75.74 | 0.002516 | 151.3M |
| ViT-B-32 LAION-2B | 0.9227 (0.9153–0.9301) | 75.64 | 0.002408 | 151.3M |
| ViT-B-16-SigLIP WebLI | 0.9515 (0.9460–0.9567) | 19.16 | 0.005004 | 203.2M |
| ConvNeXt-Base-W LAION-2B | 0.9475 (0.9416–0.9530) | 12.65 | 0.005703 | 179.4M |

**Winner:** ViT-B-16-SigLIP WebLI (`vit_b_16_siglip_webli`), with the highest
mean recall@5. **Runner-up:** ConvNeXt-Base-W LAION-2B
(`convnext_base_w_laion2b`). Both are within one percentage point of the best
recall; the winner has lower estimated energy.

The runner creates [the detailed comparison CSV](docs/model_comparison.csv),
[the four-panel comparison chart](docs/model_comparison.png),
[the recall/energy Pareto chart](docs/model_pareto.png), and
[the generated selection summary](docs/model_selection.md). Per-model
performance profiles are saved to
`data/profiles/gpu_model_profiles.json`; cached benchmark data stays under
`data/benchmark/` and is ignored by Git.

### Limitations

This is a single-dataset result on 1,000 Flickr30k test images, not evidence of
general performance across image domains. Recall confidence intervals quantify
query-sampling variability on this fixed dataset; they do not account for
training-data overlap or dataset shift. On Windows, energy is partly estimated
by CodeCarbon rather than measured directly, so energy comparisons are
approximate. CUDA inference uses fp16 autocast, while CPU inference uses fp32;
latency and energy results therefore depend on the selected hardware and
precision path. The benchmark does not evaluate caption generation or
cross-encoder reranking; see [model notes](docs/model_notes.md) for scope.

## Policy comparison results

Results below were generated by `python src/metrics/compare_policies.py` using
the checked-in synthetic traces (not real observed grid or price data), 60 jobs
per scenario, an arrival window of 120 hours, job seed 42, and a perfect
forecast (`forecast_error_std=0`, horizon 24 hours). Each policy is run against
the same jobs within its scenario. Average wait is over scheduled jobs;
deadline success is over all jobs. Negative cost/carbon changes are better than
FIFO. FIFO's average wait is zero, so a wait percentage relative to FIFO is
undefined.

### Tight slack (0.5-3.0 hours)

| Policy | Total Cost (USD) | Total Carbon (gCO2eq) | Avg Wait (hours) | Deadline Success (%) | Cost vs FIFO | Carbon vs FIFO |
|---|---:|---:|---:|---:|---:|---:|
| FIFO OnDemand | 842.25 | 84896 | 0.00 | 100.0 | +0.0% | +0.0% |
| Cheapest First | 796.17 | 100458 | 0.00 | 100.0 | -5.5% | +18.3% |
| Cleanest First | 1131.39 | 33712 | 0.00 | 100.0 | +34.3% | -60.3% |
| Joint Policy | 952.52 | 62797 | 0.00 | 100.0 | +13.1% | -26.0% |
| Cheapest-Wait | 625.36 | 131715 | 0.33 | 100.0 | -25.8% | +55.1% |
| Cleanest-Wait | 1084.90 | 33082 | 0.43 | 100.0 | +28.8% | -61.0% |
| Joint-Wait | 908.64 | 61138 | 0.40 | 100.0 | +7.9% | -28.0% |
| Cheapest-Window | 652.53 | 133697 | 0.00 | 100.0 | -22.5% | +57.5% |
| Cleanest-Window | 1131.39 | 33712 | 0.00 | 100.0 | +34.3% | -60.3% |
| Joint-Window | 926.05 | 67277 | 0.00 | 100.0 | +9.9% | -20.8% |

### Relaxed slack (2-12 hours)

| Policy | Total Cost (USD) | Total Carbon (gCO2eq) | Avg Wait (hours) | Deadline Success (%) | Cost vs FIFO | Carbon vs FIFO |
|---|---:|---:|---:|---:|---:|---:|
| FIFO OnDemand | 842.25 | 84896 | 0.00 | 100.0 | +0.0% | +0.0% |
| Cheapest First | 796.17 | 100458 | 0.00 | 100.0 | -5.5% | +18.3% |
| Cleanest First | 1131.39 | 33712 | 0.00 | 100.0 | +34.3% | -60.3% |
| Joint Policy | 952.52 | 62797 | 0.00 | 100.0 | +13.1% | -26.0% |
| Cheapest-Wait | 467.75 | 123748 | 2.82 | 100.0 | -44.5% | +45.8% |
| Cleanest-Wait | 850.36 | 30284 | 3.08 | 100.0 | +1.0% | -64.3% |
| Joint-Wait | 723.19 | 51353 | 3.15 | 100.0 | -14.1% | -39.5% |
| Cheapest-Window | 652.53 | 133697 | 0.00 | 100.0 | -22.5% | +57.5% |
| Cleanest-Window | 1131.39 | 33712 | 0.00 | 100.0 | +34.3% | -60.3% |
| Joint-Window | 926.05 | 67277 | 0.00 | 100.0 | +9.9% | -20.8% |

The tight-slack CSV and readable horizontal-bar chart are generated at
`docs/policy_comparison.csv` and `docs/policy_comparison.png`; the relaxed-slack
outputs are at `docs/policy_comparison_relaxed.csv` and
`docs/policy_comparison_relaxed.png`.

## Results: slack and forecast-error sweeps

These sweeps use **single seed, synthetic traces with a 24h cycle, perfect
forecast unless stated**; none of the trace data is real. Run
`python src/metrics/slack_sweep.py` to compare all ten policies at fixed slack
values of 1, 2, 4, 8, 12, 16, 20, and 24 hours, isolate the time-shifting
effect at slack 1, 4, 12, and 24, and evaluate Joint-Wait at 12 hours of slack
across four forecast-error levels. Forecast-error results pair the same 20 job
seeds at every level and report mean paired differences from sigma=0 with
two-sided 95% Student t intervals and mean wait hours.

Results are saved to `docs/slack_sweep.csv`, `docs/ablation_timeshift.csv`,
and `docs/forecast_error.csv`.

- [Slack sweep figure](docs/slack_sweep.png)
- [Time-shifting ablation figure](docs/ablation_timeshift.png)
- [Forecast-error sweep figure](docs/forecast_error.png)
