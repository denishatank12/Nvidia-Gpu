# GPU II: HW4.a and HW4.5

Student: Denisha Ketan Tank

Historical completed run: one Tesla T4 in Google Colab.

## Fresh RTX 4050 overnight run

Use [GPU_II_RTX4050_Run_All.ipynb](rtx4050/GPU_II_RTX4050_Run_All.ipynb) for the corrected fresh HW4.a and HW4.5 experiments. It embeds its supporting code and installs an isolated environment. Download this notebook and run it in Linux or WSL2 Jupyter on the GPU machine. Read [START_HERE.txt](rtx4050/START_HERE.txt) and [VALIDATION.txt](rtx4050/VALIDATION.txt) first.

The uploaded run exposed an oversized-batch bug: a requested batch larger than the training split was silently capped by the dataset. The new capacity probes allocate the actual requested batch and distinguish repeated-row capacity testing from unchanged-split quality training. CPU regression and reporting tests passed. Actual RTX 4050 execution still needs the real GPU smoke tests at the start of the notebook; it is not claimed verified from this Mac.

The `rtx4050` notebook is intentionally unexecuted. Save its executed copy and results ZIP after completion. The existing T4 notebook, results and `hw4-5` tag are preserved, not relabelled as new 4050 measurements. No new submission tag is created for unexecuted code.

Personal parameters: SID4 = 8834, SEED = 8834, SLICE = 834, HP_ID = 2, CLS_A = 4, CLS_B = 6.

## What to read

- [Submission notebook](GPU_II_Colab_Run_All.ipynb): the completed Colab run with all original code cells and outputs intact, plus corrected findings in clearly labelled post-run Markdown notes.
- [HW4.a findings](HW4A_METRICS.md): BERT training, evaluation, error analysis, and the first serving benchmark.
- [HW4.5 findings](METRICS.md): batch scaling, memory techniques, repeated serving, context tests, and recommendations.
- [Audit](SUBMISSION_AUDIT.md): requirement checks and the limitations that remain.
- [Checkpoint](MODEL_CHECKPOINT.md): download, file hashes, exact training command, and reload verification.
- [Run log](RUN_LOG.txt), [runtime record](RESERVATION_RECORD.md), [GPU-hour calculation](GPU_HOURS.md), and [AI-use appendix](AI_USE.md).
- [Measured outputs](outputs/): JSON and CSV measurements, plots, and both GPU UUID captures.

There is only one notebook in this folder. It contains both HW4.a and HW4.5, including the saved figures/results and corrected written findings. Some historical printed report output contains the earlier reporting errors; the post-run notes at the end supersede that interpretation. Measurements and executed outputs were not rewritten to make them appear rerun. The separate Markdown reports contain the same corrected findings.

## Data and results

Both stages used `Twitter_Data.csv` from Kaggle's `saurabhshahane/twitter-sentiment-dataset`: zero-based rows 834 through 5833, split into 3,500 training, 750 validation, and 750 test examples using SEED 8834. The recorded source hash and preparation settings match across both stages.

HW4.a test accuracy was 83.60%. The maximum-batch AMP plus 8-bit configuration was faster but reached 60.67%, so it is not an accuracy-preserving replacement. The reports separate batch-size and optimizer-update effects from independent technique comparisons.

Part C skipped (optional).

One GPU was permitted by the clarification. No second-card comparison or distributed result is claimed.

## Run again

For a new run, use the corrected [RTX 4050 notebook](rtx4050/GPU_II_RTX4050_Run_All.ipynb), not the historical T4 notebook. Hosted Colab uses its assigned GPU rather than your laptop's card. Native Windows is unsupported by vLLM; use Linux or WSL2. The old notebook's embedded code is retained as originally executed, and its post-run notes are historical rather than new GPU measurements.

The duplicate reviewed notebook and the two unexecuted local wrapper notebooks have been removed. The legacy filename `setup_rtx4090.sh` is retained for compatibility; it no longer requires a particular GPU model.

For Linux, run:

```bash
bash setup_rtx4090.sh
.venv/bin/python smoke_test.py --gpu --vllm
```

Use the combined notebook for the full experiment. Full GPU measurements must be obtained on the actual GPU, not inferred from CPU checks. The notebook builder refuses to overwrite an already executed notebook.

## Interpretation limits

The longest successful batch-8 prompt was 32,767 tokens, plus one generated token at the model's 32,768-token configured limit. No memory failure occurred within that supported range, so a GPU-memory exhaustion threshold is not claimed. The active KV-cache values are calculated; the process GPU-memory measurements also include the reserved cache pool, weights, and workspaces.

Serving latency is synchronous closed-batch completion time, not independently timestamped completion of each request. The measurements use a small number of repeated batches, so the p95 estimates are descriptive rather than a production service-level guarantee.

The audit also records repository visibility and access requirements. A tag identifies a fixed artifact snapshot; it is not a claim that those external requirements have been satisfied.
