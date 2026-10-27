# HW4.5 Metrics: Training Throughput and Memory Engineering

## Personal parameters and hardware

| Item | Value |
|---|---|
| Student | Denisha Ketan Tank |
| SID4 / SEED / SLICE | 8834 / 8834 / 834 |
| GPU | Tesla T4 |
| GPU UUID | `GPU-d3085845-9dcd-3867-c44d-22ef57457cde` |
| VRAM | 15.00 GB |
| Driver / CUDA | 580.82.07 / 12.8 |

## Dataset and fixed split

I reused **Twitter_Data.csv**, the Twitter sentiment CSV expected from HW4.a. The exact zero-based source row range was **834–5833** (Python slice `[834:5834)`). The raw file SHA-256 was `5a7af1ec5fc749ec8e5ea13c574aeb5c06254aa1c081e3421868079d5356b3f4`. The same saved 70/15/15 split and SEED 8834 were reused for every run, so none of the techniques saw different examples.

Usable examples after removing rows with a missing text or label: 5000. Split sizes: train 3500, validation 750, test 750.

## Part A: batch scaling on one Tesla T4

The clarification allowed one GPU, so I measured one Tesla T4 rather than inventing a second-card result. The full batch sweep is in `outputs/batch_scaling.csv`, and the throughput plot is `outputs/batch_size_throughput.png`.

| Physical batch | Throughput (samples/s) | Peak training memory (GB) | Time/epoch (s) | Status |
|---:|---:|---:|---:|---|
| 16 | 44.86 | 2.47 | 78.01 | success |
| 32 | 48.36 | 3.71 | 72.37 | success |
| 64 | 49.14 | 6.06 | 71.23 | success |
| 128 | 49.23 | 10.87 | 71.09 | success |
| 256 | not recorded | not recorded | not recorded | oom |

The largest full-precision batch that ran was **128**. Throughput stopped improving materially at about batch **64**, using a less-than-5% incremental gain as the cutoff. A maximum-batch ratio and VRAM ratio between two cards are not applicable because this was the approved one-GPU path.

## Part B: techniques measured independently

| Technique | Physical / effective batch | Optimizer updates | Throughput (samples/s) | Peak memory (GB) | Test accuracy |
|---|---:|---:|---:|---:|---:|---:|
| HW4.5 FP32 baseline at maximum batch | 128 / 128 | 84 | 49.60 | 10.87 | 68.00% |
| Mixed precision | 256 / 256 | 42 | 208.01 | 13.92 | 60.80% |
| Gradient checkpointing | 512 / 512 | 21 | 37.07 | 7.57 | 56.27% |
| Gradient accumulation | 16 / 128 | 84 | 48.28 | 2.89 | 66.40% |
| 8-bit optimizer | 128 / 128 | 84 | 49.63 | 10.25 | 68.00% |
| Combined candidate: AMP + 8-bit optimizer | 256 / 256 | 42 | 207.33 | 13.30 | 60.67% |

These are three-epoch runs at different batch sizes, not a controlled test of accuracy effects. A larger batch processes the same examples with fewer optimizer updates. The table therefore describes the measured speed, memory, and quality trade-off of each complete configuration; it cannot show that a technique alone caused an accuracy change.

The original HW4.a run used batch 16 and reached 83.60% test accuracy.

### Controlled one-epoch evidence

The following probes used the same physical and effective batch, examples, epoch count, and optimizer-update budget. They are separate one-epoch measurements, not the three-epoch maximum-batch runs above.

| Technique | Physical batch | Optimizer updates | Throughput (samples/s) | Peak memory (GB) | Test accuracy |
|---|---:|---:|---:|---:|---:|
| baseline | 16 | 219 | 44.86 | 2.47 | 72.27% |
| mixed_precision | 16 | 219 | 132.69 | 2.24 | 71.87% |
| gradient_checkpointing | 16 | 219 | 33.68 | 1.68 | 72.27% |
| optimizer_8bit | 16 | 219 | 44.36 | 1.87 | 72.53% |

At the common physical batch, gradient checkpointing changed throughput by **-24.9%**. That slowdown is expected because checkpointed activations are recomputed during backpropagation. Gradient accumulation used physical batch 16 × 8 steps = effective batch 128; its wall time for 84 optimizer steps was 234.41 seconds.
For accumulation, the matched direct-batch baseline used effective batch 128, 84 optimizer updates, and 10500 examples across 3 epochs. Its wall time was 228.45 seconds, compared with 234.41 seconds for accumulation. Accumulation lowers the physical batch without claiming identical predictions: dropout and floating-point reduction order can still differ.

### Memory saved per unit of throughput lost

| Rank within these controls | Technique | Matched FP32 reference | Memory saved (GB) | Throughput change (samples/s) | GB saved per sample/s lost |
|---:|---|---|---:|---:|---:|
| 1 | mixed_precision | physical batch 16, 1 epoch | 0.22 | 87.83 | no throughput loss |
| 2 | gradient_accumulation | effective batch 128, 3 epochs | 7.98 | -1.32 | 6.0311 |
| 3 | optimizer_8bit | physical batch 16, 1 epoch | 0.60 | -0.51 | 1.1871 |
| 4 | gradient_checkpointing | physical batch 16, 1 epoch | 0.79 | -11.19 | 0.0703 |

The ranking includes all four independent techniques and excludes the combined candidate. AMP, checkpointing, and the 8-bit optimizer use the one-epoch common-physical-batch control. Accumulation uses the three-epoch matched-effective-batch control described above. This is a descriptive ranking against the stated references, not a universal ranking across identical workloads. Positive throughput change means faster; a speedup with memory savings has no finite memory-per-throughput-loss ratio. These short single-run differences include timing noise.

I would apply mixed precision first because the 4090 has fast tensor cores and AMP normally improves both memory use and speed. I would add the 8-bit optimizer when optimizer state is the limiting memory. On a 32 GB card I would not use checkpointing for this BERT run unless a required batch still failed to fit, because its recomputation cost is measurable.

## Part C

**Part C skipped (optional)**

## Part D: serving on the Tesla T4

Model: `Qwen/Qwen2.5-0.5B-Instruct` at float16; 64 maximum generated tokens per request. Latency means Closed-batch wall time from synchronous submission until every response in the batch is returned.

| Configuration | Tokens/sec | p50 latency (ms) | p95 latency (ms) |
|---|---:|---:|---:|
| HF pipeline, batch 1 | 29.95 | 2127.67 | 2267.57 |
| vLLM, batch 1 | 66.10 | 951.88 | 1063.99 |
| vLLM, batch 8 | 503.68 | 1022.93 | 1030.10 |
| vLLM, batch 32 | 1606.64 | 1240.24 | 1391.27 |

None of the tested configurations showed worse p50 latency than the HF baseline while also increasing throughput over the preceding configuration. Batching can increase total work completed per second while individual requests wait longer. For an interactive application, p50 and especially p95 latency matter because a user waits for a response. For an offline job, aggregate tokens per second is usually more important.

The measured longest successful context at batch 8 was **32767 tokens per request**. `outputs/kv_cache_context.csv` gives both the active KV-cache requirement and the GPU memory reported around each request. The active-cache calculation uses the model's actual layer, KV-head, head-dimension, and FP16 byte counts. vLLM reserves a cache pool when its engine starts, so total process memory can stay nearly flat while the amount of that pool occupied by active tokens grows.
The configured model context limit was 32768 tokens, including the generated token. Every tested length succeeded, so no GPU-memory exhaustion boundary was observed. The largest successful prompt is a tested lower bound on supported prompt length, not a measured memory wall. Extending past the configured model limit would test a different configuration. The KV-cache column is calculated active storage, not an isolated measurement of live KV tensors. The measured GPU-used column also includes the reserved pool, model weights, and workspaces.

| Batch | Context length | Calculated active KV cache (GB) | Measured GPU used after request (GB) | Result |
|---:|---:|---:|---:|---|
| 8 | 512 | 0.0469 | 11.9023 | success |
| 8 | 1024 | 0.0938 | 12.2070 | success |
| 8 | 2048 | 0.1875 | 12.6523 | success |
| 8 | 4096 | 0.3750 | 13.9023 | success |
| 8 | 8192 | 0.7500 | 14.1211 | success |
| 8 | 16384 | 1.5000 | 14.1211 | success |
| 8 | 32767 | 2.9999 | 14.1211 | success |

### Comparison with my HW4.a numbers

Compared with my recorded HW4.a training time, the HW4.5 maximum-batch baseline changed by -6.7%. However, the batch changed from 16 to 128, so this is not a hardware-only speed comparison. Both serving stages used the same recorded GPU, FP16 precision, and closed-batch latency definition. The small serving differences do not establish a hardware improvement; timing variability may contribute. The exact earlier values remain in `hw4a_baseline.json`.

| Configuration | HW4.a tokens/s | HW4.5 tokens/s | HW4.a p50 (ms) | HW4.5 p50 (ms) | HW4.a p95 (ms) | HW4.5 p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| HF pipeline, batch 1 | 28.53 | 29.95 | 2114.15 | 2127.67 | 2823.26 | 2267.57 |
| vLLM, batch 1 | 66.08 | 66.10 | 949.92 | 951.88 | 1037.75 | 1063.99 |
| vLLM, batch 8 | 478.11 | 503.68 | 1066.14 | 1022.93 | 1107.22 | 1030.10 |
| vLLM, batch 32 | 1631.81 | 1606.64 | 1249.70 | 1240.24 | 1270.21 | 1391.27 |

## Part E: recommendation and GPU-hours

The tested combined candidate was AMP plus the 8-bit optimizer: peak allocated memory 13.30 GB, throughput 207.33 samples/s, and measured training wall time 67.10 seconds. This three-epoch run used **0.0186 GPU-hours**: one GPU × 67.10 seconds ÷ 3600. This is not the GPU cost of the whole benchmark suite.

The combined candidate reached 60.67% test accuracy; AMP alone reached 60.80% at 208.01 samples/s. The combination used 0.62 GB less peak memory than AMP alone at the same batch, but was not the fastest or most accurate configuration. The original HW4.a configuration achieved 83.60% test accuracy.

For accuracy in these measured runs, I would keep the original HW4.a configuration. If memory and speed matter more, I would test AMP at that same batch and optimizer-update budget before replacing it. AMP plus the 8-bit optimizer is useful when the additional measured memory saving matters, but the current maximum-batch run does not preserve the original accuracy. This recommendation uses validation for future configuration selection; the reported test results are descriptive, not a reason to tune repeatedly on the test set.

## Baseline class behavior

Validation accuracy was best at epoch **3**. The final epoch was the best one, so this run did not train past the improvement point. The worst test class by recall was **-1.0**. Per-class precision/recall and the confusion matrix are saved in `outputs/baseline_per_class_metrics.csv` and `outputs/baseline_confusion_matrix.png`.

| Epoch | Training loss | Validation loss | Validation accuracy | Training seconds |
|---:|---:|---:|---:|---:|
| 1 | 1.0098 | 0.9433 | 55.47% | 71.04 |
| 2 | 0.8632 | 0.8424 | 64.67% | 70.52 |
| 3 | 0.6866 | 0.7696 | 68.67% | 70.12 |

| Class | Precision | Recall | Support |
|---|---:|---:|---:|
| -1.0 | 0.5460 | 0.5329 | 167 |
| 0.0 | 0.8204 | 0.6259 | 270 |
| 1.0 | 0.6614 | 0.8051 | 313 |

### Five misclassified examples from the worst class

| Source row | True | Predicted | Tweet | Diagnosis |
|---:|---|---|---|---|
| 2008 | -1.0 | 1.0 | ‘chowkidaar mentality hai’ the insinuation that narendra modi has the crookcheap mentality regular chowkidaar respect for ordinary hardworking indians | The phrase crookcheap mentality is an insult, while respect for ordinary hardworking Indians is positive wording; the positive prediction may over-weight that praise and miss the insult. |
| 4206 | -1.0 | 1.0 | already have one his name narendra damodardas modi common man who came power with hard work alone india stands his side | The visible text praises a leader rising through hard work and says India stands by him; the positive prediction is reasonable from this wording, so the negative label may be noisy or depend on missing context. |
| 4772 | -1.0 | 1.0 | leaving aside all the hate against dynastic politics think this must have step for uplifting people living brink just want congress government for this one thing hope modi comes with such yojna would love seeing someone getting 6kmonth | The tweet mentions hate but also praises a welfare proposal and expresses hope and support; the mixed wording makes a single sentiment label difficult, and the positive output agrees with several explicit cues. |
| 1667 | -1.0 | 1.0 | ratan naval tata family patriarch the billion tata empire has lavished praise prime minister narendra modi underscoring that his leadership what the country needsat this point time and urging citizens rally round his vision create new india | The text explicitly praises Modi and his leadership, so the positive prediction fits the visible wording; the negative dataset label may be noisy, although the missing original context prevents certainty. |
| 5542 | -1.0 | 1.0 | game changer modi era finished | Modi era finished expresses a negative judgment in a short tweet; the phrase game changer may have supplied a competing positive cue, although this is only a possible explanation. |

The same examples, including their diagnoses, are saved in `outputs/baseline_misclassified_examples.csv`.

## HW4.5 summary

Combined speed/memory candidate: **AMP + 8-bit optimizer** — 13.30 GB peak allocated, 207.33 samples/s, 67.10 seconds, 60.67% test accuracy. It is not an overall winner because the original HW4.a configuration had higher accuracy. This is the HW4.5 summary and is not an HW5 baseline.
