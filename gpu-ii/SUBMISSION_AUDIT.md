# HW4.a and HW4.5 audit

Audit date: October 27, 2026. Student: Denisha Ketan Tank.

## Verdict

The original combined Colab notebook completed all nine code cells without an uncaught notebook error. The required experiments produced measurements and the saved BERT model reloads successfully. The reviewed reports correct the hardware, comparison controls, ranking, and error analysis without changing measurements.

This is not an unconditional claim that every administrative requirement is satisfied. At audit time, GitHub reported this repository as public. The standing instructions require a private repository shared with the instructor and grading team. Visibility and grader access need confirmation before submission.

## Requirement checks

| Requirement | Evidence and result |
|---|---|
| Parameters and initialization | SID4/SEED 8834, SLICE 834, HP_ID 2, CLS_A 4, CLS_B 6 are recorded; initialization and split use the saved seed. |
| Same Twitter data for both assignments | Source rows 834–5833; identical source hash and preparation settings; 3,500/750/750 split. |
| BERT preprocessing and training | BERT tokenizer, TensorDatasets, and BertForSequenceClassification; three completed epochs, saved model and tokenizer. |
| Losses, validation, time, and memory | Per-epoch CSV and loss plots; HW4.a test accuracy 83.60%, wall time 244.93 s, peak allocated memory 2.467 GiB. Best validation epoch was the last, so no later-epoch waste is claimed. |
| Class analysis | Confusion matrix, per-class precision/recall, and five individually discussed errors from the lowest-recall class for HW4.a and the HW4.5 baseline. Possible label noise is identified cautiously. |
| Save/reload custom inference | Two recorded predictions; an independent CPU reload reproduced class IDs [2, 0]. Checkpoint files and hashes are documented. |
| HF versus vLLM serving | Both stages recorded HF batch 1 and vLLM batches 1, 8, 32 for Qwen/Qwen2.5-0.5B-Instruct, with token throughput, p50, and p95. |
| Batch scaling and first OOM | FP32 batches 16, 32, 64, 128 succeeded; 256 failed with CUDA OOM. Throughput plateau identified near batch 64. The expected OOM is evidence, not a failed notebook run. |
| Independent techniques and combination | Separate AMP, checkpointing, accumulation, and 8-bit results, followed by the combined candidate. Max-batch accuracy comparisons are labelled as confounded by different optimizer-update counts. |
| Ranking | All four independent techniques included. Three techniques use common-batch one-epoch controls; accumulation uses the matched effective-batch three-epoch control. The reference difference and timing noise are disclosed. |
| Accumulation budget | Physical batch 16 × 8, effective batch 128; 84 optimizer updates and 10,500 samples match the direct-batch baseline. |
| Part C | Part C skipped (optional). |
| One-GPU clarification | Actual Tesla T4 and UUID captured before and after; no fabricated second-card ratio or distributed benchmark. |
| Context sweep | Batch 8 tested prompts from 512 to 32,767 tokens, all successful. This reaches the configured 32,768-token limit with one generated token, not a demonstrated GPU-memory exhaustion boundary. |
| GPU-hours and recommendation | Original accuracy-oriented run: 0.0680 GPU-hours; faster, lower-accuracy combined candidate: 0.0186 GPU-hours. Neither is presented as the cost of the entire lab. |
| Standing artifacts | One executed notebook with corrected post-run analysis, figures, measured CSV/JSON, timestamped logs, AI-use appendix, hosted-runtime record, GPU-hour record, UUID captures, and checkpoint download instructions are included. |

## Limits that should remain visible

1. The serving code assigns the synchronous batch-completion time to each request. It does not record each request's first-token or independent finish time. The p50/p95 values have this stated definition and come from a small number of batch repetitions.
2. The context experiment did not exhaust GPU memory within the supported model range. Calculated active KV storage and measured total process GPU usage are different quantities. A literal memory-wall number or isolated live-KV measurement would require a further GPU experiment; neither can be invented from this run.
3. The max-batch runs have different numbers of optimizer updates. The common-batch probes provide controlled one-epoch evidence, but they do not establish matched-budget three-epoch quality equivalence for every technique.
4. No GPU experiment was rerun during the post-run audit. CPU reload, artifact checks, arithmetic checks, and report regeneration do not establish fresh GPU timings.
5. Original executed code cells and outputs are preserved as historical run evidence. Corrections are included in clearly labelled post-run Markdown cells in the same notebook and in the separate reports, not disguised as a new execution. The duplicate unexecuted notebooks were removed.

## Submission and access

The intended artifact tag is `hw4-5`. The large original results archive is attached to the corresponding GitHub release, with the checkpoint location and hashes in `MODEL_CHECKPOINT.md`. A release or tag identifies the submitted snapshot; it does not resolve the public/private repository requirement or confirm that the instructor has access. Verify those access requirements before submitting the link.
