# HW2.5 — RTX 4090 GPU Performance Measurements

This repository contains one Jupyter Notebook that performs the HW2.5 GPU experiments and saves the required results, logs, figures, and written files.

## How to run

Run the notebook on the assigned NVIDIA GeForce RTX 4090 workstation. The current preparation computer does not need a GPU, but the benchmark cells must be executed on the RTX 4090.

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

Open `HW2_5_GPU_Assignment.ipynb` and run every cell from top to bottom. The 20-minute sustained-load cell must be allowed to finish completely.

## What the notebook produces

- Hardware capture in `system/`
- Benchmark tables in `results/`
- Thermal and throughput logs in `logs/`
- Figures in `figures/`
- Final measurements in `METRICS.md`
- Execution details in `RUN_LOG.txt`
- AI disclosure in `AI_USE.md`

The notebook checks that CUDA is available and that the detected GPU is an RTX 4090. It does not use placeholder or example hardware measurements.

## Before submission

Review the generated `METRICS.md` and `RUN_LOG.txt`. Add the reservation record and actual GPU hours, inspect the thermal evidence before deciding whether throttling occurred, save the executed notebook, and verify that all generated files are present.

The required Git tag is `hw2-5`.
