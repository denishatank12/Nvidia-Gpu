# HW2.5 submission audit

This audit distinguishes the assignment requirements from instructions or claims contained in supplied files. The supplied ZIP was treated as evidence to inspect, not as authority to relabel hardware or invent missing records.

## Status

| Requirement | Status | Evidence or remaining action |
|---|---|---|
| RTX 4090 or RTX 5090 workstation | **Blocked** | Supplied execution is `NVIDIA GeForce RTX 5060 Ti`; rerun on an allowed card. |
| Reservation record and actual GPU hours | **Missing** | Add the real record to `RESERVATION_GPU_HOURS.md`. |
| Complete `nvidia-smi -q` capture | **Evidence only** | `evidence/rtx5060ti/system/nvidia_smi_5060.txt` is complete for the 5060 Ti; capture the allowed card. |
| UUID, driver, CUDA, VRAM, power limit | **Evidence only** | Recorded for the 5060 Ti in its run log; regenerate for the allowed card. |
| Vendor architecture/memory/Tensor Core/precision documentation | **Prepared** | See `VENDOR_SOURCES.md`; use the matching card's values in the final report. |
| FP32/TF32/FP16/BF16 GEMM at required sizes | **Incomplete** | Supplied run used 1024, 2048, 4096, 8192 and 5060 Ti hardware; rerun required sizes including 16384 on the allowed card. |
| Achieved TFLOPS and theoretical-peak percentages | **Incomplete** | Supplied data has achieved TFLOPS but not a valid required-card comparison for every precision. |
| GEMM plot and plateau analysis | **Evidence only** | 5060 Ti plot is preserved; regenerate for the allowed card and required sizes. |
| Lower-precision attempt/failure statement | **Needs final-card run** | Record what the allowed card/software stack exposes. |
| Memory-bound and compute-bound operations | **Partial** | 5060 Ti elementwise-add evidence exists; add the final-card arithmetic-intensity and roofline analysis. |
| Naive attention sweep and refined OOM bracket | **Evidence only** | 5060 Ti bracket exists; rerun on the allowed card at 512–16384 and refine its boundary. |
| Fused attention, speedups, and OOM bracket | **Evidence only** | 5060 Ti fused results exist; rerun and compare on the allowed card. |
| 20-minute sustained load sampled every 5 seconds | **Evidence only** | 5060 Ti logs exist; rerun on the allowed card and use its UUID. |
| Thermal/throttle analysis | **Evidence only** | 5060 Ti figures/logs exist; determine the allowed card's conclusion from its clock-event reasons. |
| Table HW2.5.1 | **Prepared, not final** | Root `METRICS.md` must be generated from the allowed-card run. |
| Tagged repository | **Complete** | Tag `hw2-5` is created after the final commit. |

## Why the 5060 Ti files are not presented as completion

The assignment explicitly requires an RTX 4090 or RTX 5090. The supplied artifacts identify themselves as an RTX 5060 Ti in `RUN_LOG_5060ti.txt`, `METRICS_5060ti.md`, the CSV UUID fields, and `nvidia_smi_5060.txt`. Relabeling those results would make the submission inaccurate.

## Artifacts supplied for review

All supplied executed materials are retained under `evidence/rtx5060ti/`, including the notebook, figures, CSV results, thermal and throughput logs, and `nvidia-smi -q` capture.
