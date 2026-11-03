# GPU II: HW4.a and HW4.5

Student: Denisha Ketan Tank.

## Submission

Use the [executed T4 notebook](GPU_II_Colab_Run_All.ipynb). All ten code cells completed in order on October 28, 2026, with no saved error tracebacks. It contains both HW4.a and HW4.5, including the measured results and figures. The reviewed findings are also available as the two Markdown reports below.

The November 3 cleanup keeps one submission notebook. It removes the RTX folder, duplicate unexecuted T4 template and old setup wrappers. The executed code, counts, outputs and raw measurements are unchanged. The final Markdown section in the notebook contains the same clearer report wording as the standalone findings files.

[Download the original results archive and full checkpoint](https://github.com/denishatank12/Nvidia-Gpu/releases/download/hw4-5/GPU_II_T4_RESULTS_6ec28b7ced2a.zip).

The submission snapshot is the `hw4-5` tag. The release also keeps the original executed notebook and results archive, before the readability edits. Choose the archive with run ID `6ec28b7ced2a`. The older `GPU_II_RESULTS.zip` asset is historical and is not needed for this submission.

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

The [course clarification supplied with this task](COURSE_CLARIFICATIONS.md) records the permitted skips, fixed-data requirements and interpretation of the observed serving-capacity boundary. It does not claim a separate physical-VRAM stress test or waive the repository-access requirement.

## What is kept for reproduction

The executed notebook is self-contained and embeds the source and setup used for the run. To repeat it, open a copy in a Linux CUDA notebook environment, select a T4 in Colab and choose Run all. Repeating the run is not needed to inspect the saved submission.

The readable Python modules, tests, locked requirements, installed versions, settings and data manifest remain in this folder. outputs/ contains the raw measurements, figures and capacity-probe logs. RUN_LOG.txt is the complete run log; the removed HW4.a log was an identical copy. The full classifier and tokenizer are in the linked results archive rather than a large Git blob. No measured result was changed during cleanup.

## Access qualification

Repository visibility is unchanged at the user's request. It remains public, although the standing course instructions request a private repository shared with the instructor and graders. Publication and tagging do not resolve that discrepancy or certify the student's personal understanding.
