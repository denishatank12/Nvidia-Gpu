# GPU II: HW4.a and HW4.5

Student: Denisha Ketan Tank

Actual completed run: one Tesla T4 in Google Colab.

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

For a fresh Colab run, open [GPU_II_Colab_Run_All.ipynb](GPU_II_Colab_Run_All.ipynb), choose a GPU runtime, and select Run all. It installs an isolated Python 3.11 environment and downloads the public dataset and model files. The runtime must provide a supported NVIDIA GPU; a local Mac CPU is not sufficient for vLLM CUDA benchmarks. Its embedded code is retained as originally executed. The post-run notes describe the October 27 run; they are not regenerated measurements for a future GPU session. The repository's finalizer scripts contain the reporting corrections.

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
