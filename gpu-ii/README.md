# GPU II: HW4.a and HW4.5

Student: Denisha Ketan Tank.

## Submission

Use the [executed notebook](GPU_II_Colab_Run_All.ipynb). All ten code cells completed in order on October 28, 2026, on one Tesla T4. The notebook contains the measured results, figures and generated reports. Clearly labelled post-run notes clarify the FP32 throughput finding and replace the original AI-use draft without changing any executed code or output.

[Download the original results archive and full checkpoint](https://github.com/denishatank12/Nvidia-Gpu/releases/download/hw4-5/GPU_II_T4_RESULTS_6ec28b7ced2a.zip).

The final submission snapshot is the `hw4-5` tag. The release also preserves the original executed notebook and the earlier results archive as immutable evidence. The new archive is the file with run ID `6ec28b7ced2a`; the older `GPU_II_RESULTS.zip` is historical, not the current run.

## Findings and evidence

- [HW4.a findings](HW4A_METRICS.md): BERT results, class analysis, saved-model inference and the first serving benchmark.
- [HW4.5 findings](METRICS.md): real batch-size memory walls, independent techniques, matched-update controls, repeated serving, observed KV-cache capacity and GPU-hours.
- [Submission audit](SUBMISSION_AUDIT.md): independent checks and remaining qualifications.
- [AI-use appendix](AI_USE.md), [checkpoint and hashes](MODEL_CHECKPOINT.md), [GPU-hour calculation](GPU_HOURS.md), and [runtime record](RESERVATION_RECORD.md).
- [Measured outputs](outputs/) and [timestamped run log](RUN_LOG.txt).

Both assignments use Kaggle's `saurabhshahane/twitter-sentiment-dataset`, source rows 834 through 5833 inclusive. The same SEED 8834 split contains 3,500 training, 750 validation and 750 test examples.

The original batch-16 BERT fine-tune reached 83.60% test accuracy. It took 255.65 seconds including validation and used 2.466 GiB peak allocated training memory. All six controlled conditions completed three epochs, 10,500 training examples and 657 optimizer updates. Maximum-batch quality results are kept separate because those runs have fewer updates.

FP32 throughput was highest at batch 16, at 50.05 samples/s. Larger tested batches did not improve it. Batch 128 was the largest successful FP32 batch; batch 256 produced the expected CUDA OOM.

The context experiment verified eight concurrent prompts of 29,296 tokens each. At 29,424 tokens, the configured cache could no longer admit all eight together. This is a 128-token engine-configuration capacity bracket, not a universal hardware-only or exact one-token OOM limit.

Part C skipped (optional). One GPU is permitted by the clarification; no second-card comparison or distributed result is invented.

## Reproducing the run

For a future T4 run, use [the self-contained T4 run notebook](t4/GPU_II_T4_Run_All.ipynb) and its [instructions](t4/START_HERE.txt). For a local RTX 4050, use the separate [RTX 4050 notebook](rtx4050/GPU_II_RTX4050_Run_All.ipynb). These templates are distinct from the executed submission notebook above.

The Python source, locked requirements, installed package versions, manifest and raw results from the actual T4 run are included in the archive. The executed notebook embeds that source and performs its own setup; no legacy wrapper is needed. The revised AI-use and plateau wording are post-run editorial changes, not new GPU measurements.

## Access qualification

Repository visibility is unchanged at the user's request. It remains public, although the standing course instructions request a private repository shared with the instructor and graders. Publication and tagging do not resolve that discrepancy or certify the student's personal understanding.
