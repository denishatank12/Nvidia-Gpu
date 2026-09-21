# HW2.5 submission audit

The instructor approved completing this lab on the student's personal laptop GPU. The executed measurements in this repository therefore describe the NVIDIA GeForce RTX 5060 Ti identified in the captured `nvidia-smi -q` output and UUID-labelled result files.

| Requirement | Status | Evidence |
|---|---|---|
| Approved GPU run | Complete | RTX 5060 Ti; see `system/nvidia_smi_5060.txt`. |
| Reservation and GPU-hour record | Not applicable | Personal-laptop run; documented in `RESERVATION_GPU_HOURS.md`. |
| UUID, driver, CUDA, VRAM, power limit | Complete | `RUN_LOG.txt` and `system/nvidia_smi_5060.txt`. |
| Vendor architecture/memory/Tensor Core/precision documentation | Complete | `VENDOR_SOURCES.md`. |
| GEMM throughput measurements | Complete | `results/matmul_results.csv` and `figures/matmul_tflops.png`. |
| Effective bandwidth and roofline discussion | Complete | `results/bandwidth_results.csv` and `METRICS.md`. |
| Naive attention sweep and refined OOM bracket | Complete | `results/attention_naive*.csv`; largest success 4992 and smallest failure 5120. |
| Fused attention measurements | Complete | `results/attention_fused.csv`; no OOM in the tested range. |
| Sustained 20-minute load and 5-second sampling | Complete | `logs/thermal_5060.csv` and `logs/throughput_5060.csv`. |
| Figures and summary table HW2.5.1 | Complete | `figures/` and `METRICS.md`. |
| AI disclosure | Complete | `AI_USE.md`. |
| Tagged repository | Complete | Git tag `hw2-5`. |

The `evidence/rtx5060ti/` directory preserves the original supplied bundle for provenance; the required deliverables are also available at the repository root.
