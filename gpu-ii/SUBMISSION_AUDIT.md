# HW4.a and HW4.5 audit

Audit date: October 28, 2026. Student: Denisha Ketan Tank.

## Verdict

The supplied T4 notebook completed all ten code cells in order with no saved error tracebacks. The original ZIP passed its CRC check. Independent checks reproduced the source slice and split, recalculated reported statistics and reloaded the actual checkpoint. No full GPU rerun was needed for the publication edits.

The notebook's executed code cells and outputs are preserved exactly. Post-run Markdown notes and reviewed repository reports clarify the FP32 throughput maximum and the AI-use appendix. The original ZIP and original executed notebook are retained as release assets, so the original evidence can be compared with those editorial revisions.

## Requirement checks

| Requirement | Evidence and result |
|---|---|
| Dataset and split | CSV hash verified. All 5,000 source texts/labels match rows 834–5833. Retokenization reproduces the saved tensors. SEED 8834 independently reproduces the disjoint 3,500/750/750 split. |
| BERT training | Three full epochs; 10,500 training examples; 657 baseline optimizer updates. Per-epoch losses and validation accuracy are saved. |
| Quality, time and memory | Test accuracy 83.60%; 255.65 seconds including validation; 2.466 GiB peak allocated training memory. Best validation accuracy occurs at epoch 3, with no observed later training waste. |
| Class analysis | Confusion matrix, precision/recall and five lowest-recall-class errors are present. Their source rows belong to the test split, and the saved checkpoint reproduces all five predictions. |
| Save/reload | Full classifier and tokenizer are in the release archive. An independent offline CPU reload reproduced both custom predictions and their reported confidence within tolerance. |
| Batch scaling | Real FP32 batches 16/32/64/128 succeed; 256 produces CUDA OOM. Throughput is highest at batch 16 and declines at every tested increase. |
| Memory techniques | AMP, checkpointing, 8-bit optimizer and a combined candidate each have independent full-epoch sweeps to a genuine recorded CUDA OOM, followed by three-epoch quality runs. |
| Fair comparisons | All six matched controls use effective batch 16, three epochs, 10,500 examples and 657 completed updates, with no skipped updates. Accumulation uses physical batch 4 times 4 microbatches. |
| Ranking and recommendation | All four independent techniques are ranked. The combined candidate is kept separate. The selected baseline is the fastest candidate meeting the stated validation-accuracy rule; this selection was independently recomputed. |
| Part C and one GPU | Part C skipped (optional). The actual T4 UUID is captured before and after. The clarification permits one GPU. |
| Serving | HF batch 1 and vLLM batches 1/8/32 are measured in both stages. Request counts, 64-token lengths, throughput, p50/p95 and individual completion-timestamp arithmetic recompute correctly from the raw CSVs. |
| KV cache | Actual tensor-storage and allocated-block arithmetic verify. Eight concurrent prompts succeed at 29,296 tokens; the configured pool cannot admit all eight at 29,424. The bracket is 128 tokens wide. |
| GPU-hours | One full baseline fine-tune uses 255.648288 / 3600 = 0.071013 GPU-hours. This is not the cost of the entire experimental session. |
| Artifacts | Executed notebook, plots, raw CSV/JSON, logs, AI-use appendix, hosted-runtime record, GPU-hour record, UUID captures and full-checkpoint download are included. |

## Interpretation and access qualifications

1. Maximum-batch runs have different optimizer-update budgets and lower accuracies. The separate matched-budget controls address that comparison; large-batch differences are not attributed solely to a memory technique.
2. Serving requests within a synchronous batch can be correlated. Raw individual request timestamps are available, but measured p95 is not a production service-level guarantee.
3. The context threshold is an observed eight-request cache-admission limit under the recorded engine settings. It is not a claim of universal physical-VRAM exhaustion or an exact one-token threshold.
4. The student ran the GPU experiments. The assistant performed the independent artifact/code audit, CPU checkpoint checks and editorial revisions. These activities are distinguished in AI_USE.md; personal understanding is not certified by an automated check.
5. At publication time the repository remains public because the user explicitly asked to leave visibility unchanged. The standing private-repository requirement is therefore unresolved, not silently waived.

The final submission reference is `hw4-5`. The original archive and notebook are release assets. Checkpoint files and hashes are documented in MODEL_CHECKPOINT.md.
