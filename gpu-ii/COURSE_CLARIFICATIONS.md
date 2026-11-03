# GPU II requirement scope

This note records the course clarification supplied in the assignment conversation. It is not a new instructor approval or an independently retrieved Canvas announcement.

## Permitted skips

The supplied clarification says that Part C, using two GPUs at once, is optional. A skipped submission must include the exact statement:

> Part C skipped (optional)

It also says that one GPU is enough and permits lab hardware, Colab, Kaggle or a personal GPU. The selected submission uses one Tesla T4. It does not claim two-card throughput plots, maximum-batch ratios, VRAM ratios or distributed-training results. This scope follows the supplied one-GPU clarification rather than presenting missing second-card data as measured.

## Data that must remain unchanged

The clarification requires the same Twitter dataset, consecutive source rows and fixed split for HW4.a and HW4.5. The submitted T4 run uses Kaggle's saurabhshahane/twitter-sentiment-dataset, zero-based rows 834 through 5833 inclusive, SEED 8834 and the same saved 3,500/750/750 train, validation and test indices throughout.

## Meaning of the serving-capacity result

Part D asks for KV-cache memory versus context length and the longest context served at batch eight before exhausting memory. Unlike Part A, it does not explicitly require a CUDA out-of-memory exception.

The actual T4 experiment measures the capacity of the configured serving engine. All eight full prompts fit together at 29,296 tokens per request. At 29,424 tokens they no longer fit together in its configured cache. Both the cache storage and scheduler occupancy were observed. The successful/failing bracket is 128 tokens wide.

This is a measured eight-request cache-admission boundary, not an exact one-token threshold or a universal physical-VRAM limit. The report preserves that distinction. A physical-memory stress test, a different cache budget or another GPU was not performed and is not claimed.

## Submission status

HW4.a, the single-GPU Part A and Part B experiments, repeated serving, context-capacity measurements and Part E analysis are included. Part C is explicitly skipped. The executed notebook, raw measurements, figures, logs, UUID captures, checkpoint archive, runtime record, GPU-hour record and AI-use appendix are published under the existing hw4-5 submission reference.

The standing private-repository instruction remains separate from those technical results. This repository is still public under the student's earlier visibility preference. The clarification above does not waive that requirement. Visibility and grader access must be resolved before claiming every administrative requirement is satisfied.
