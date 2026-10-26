# DATA 266 HW4.5: GPU Training Throughput and Memory Engineering

Student: Denisha Ketan Tank  
GPU path: one NVIDIA GeForce RTX 4090  
Personal parameters: SID4 = 8834, SEED = 8834, SLICE = 834, HP_ID = 2, CLS_A = 4, CLS_B = 6

This folder contains both HW4.a and the one-GPU HW4.5 continuation. Part C is optional and is intentionally skipped. HW4.a creates the BERT baseline and vLLM table. HW4.5 then keeps the BERT model, Twitter rows, frozen 70/15/15 split, seed, sequence length, learning rate, and epoch count fixed while it changes one memory technique at a time.

## Before running

`Homework_4A.ipynb` downloads the public `saurabhshahane/twitter-sentiment-dataset` from Kaggle when `Twitter_Data.csv` is absent. The default columns are `clean_text` and `category`. Its finalizer fills `hw4a_baseline.json` automatically, so no old value is copied by hand or invented.

The code selects zero-based rows 834 through 5833 when at least 5,834 rows exist. It stores the source-file SHA-256 hash, exact row numbers, excluded missing rows, label mapping, and split indices in `cache/dataset_manifest.json`.

## Fast RTX 4090 run

```bash
bash setup_rtx4090.sh
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Homework_4A.ipynb \
  --ExecutePreprocessor.timeout=-1
.venv/bin/jupyter nbconvert --to notebook --execute --inplace Homework_4_5.ipynb \
  --ExecutePreprocessor.timeout=-1
```

The notebook calls the benchmark scripts and creates all measured tables, plots, logs, UUID captures, `METRICS.md`, and the GPU-hour calculation. It uses only GPU 0.

## Efficiency choices

- Twitter examples are tokenized once and reused from a tensor cache.
- Every OOM probe uses a new process, preventing a failed batch from fragmenting later runs.
- Pinned-memory data loading, nonblocking copies, TF32, and fused AdamW are enabled where supported.
- AMP uses FP16 autocast and gradient scaling.
- vLLM runs offline closed-batch measurements at batch sizes 1, 8, and 32.
- Context tests generate one token, which measures capacity without wasting generation time.

## Files produced after the run

- `HW4A_METRICS.md`: original BERT and serving results
- `hw4a_baseline.json`: measured HW4.a values consumed by HW4.5
- `outputs/hw4a_*`: HW4.a tables, figures, saved/reload proof, and serving results
- `MODEL_CHECKPOINT.md`: checkpoint location, reproduction command, and final large-file link field
- `METRICS.md`: final tables and analysis, including the exact sentence `Part C skipped (optional)`
- `RUN_LOG.txt`: timestamped command output
- `outputs/batch_scaling.csv`: every successful batch and the first OOM
- `outputs/technique_comparison.csv`: required memory-technique table
- `outputs/serving_benchmark.csv`: HF and vLLM throughput/latency
- `outputs/kv_cache_context.csv`: batch-8 context capacity and calculated KV-cache size
- `outputs/gpu_uuid_before.txt` and `outputs/gpu_uuid_after.txt`: the two UUID captures
- `GPU_HOURS.md`: explicit GPU-hour calculation

The `checkpoints/` and `cache/` folders stay local because the BERT weights and tokenized dataset are large and reproducible.

## Two manual items before submission

The code can create every measured result, but it cannot know the details of a lab reservation or upload a large checkpoint to an approved destination automatically. Fill every field in `RESERVATION_RECORD.md`, upload `checkpoints/hw4a_bert_classifier/` to approved cloud storage or Git LFS, and replace `pending upload` in `MODEL_CHECKPOINT.md` with the shareable link. The final notebook check deliberately fails until both items are complete.
