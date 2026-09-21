# HW2.5 GPU Performance Measurements

GPU: NVIDIA GeForce RTX 5060 Ti
GPU UUID: GPU-af815c34-e868-2ca8-0057-81c8ac28573e

## Table HW2.5.1

| Measurement | NVIDIA GeForce RTX 5060 Ti | Notes |
|---|---:|---|
| Peak achieved TFLOPS (BF16) | 47.280220 | Best measured BF16 GEMM result |
| % theoretical peak (BF16) | Not reported because no directly comparable official BF16 TFLOPS reference is published | The exact desktop or laptop reference must match the measured GPU variant |
| Effective bandwidth (GB/s) | 345.532936 | FP32 elementwise add |
| Naive attention OOM length | largest tested success: 4992; smallest tested failure: 5120 | Tested bracket, not an invented exact limit |
| Fused attention OOM length | largest tested success: 8192; no OOM observed in tested range | Tested bracket, not an invented exact limit |
| Steady state / peak throughput | 96.61% | Final 5 minutes / first 30 seconds peak |
| Throttle assessment | No clear sustained throttling signature from the conservative clock check | Review the thermal graph and NVIDIA clock-event reasons before submission |

## Analysis

### Precision

Throughput should be interpreted together with matrix size. Small matrices may underutilize the GPU because launch and scheduling overheads are large relative to the available parallel work. FP16 and BF16 can use the RTX 5060 family Tensor Cores, while FP32 and TF32 use different arithmetic paths. At larger sizes, the measured curves show whether throughput approaches a plateau.

### Bandwidth

The elementwise addition has low arithmetic intensity: each output requires little computation but multiple values must be read or written. Its performance is therefore primarily constrained by memory movement. Matrix multiplication reuses matrix values many times and has much higher arithmetic intensity, so it is more compute-oriented.

### Attention

Naive attention stores a sequence-by-sequence score matrix, so its memory usage grows approximately quadratically with sequence length. Fused attention avoids materializing that complete matrix in global memory, reducing peak VRAM use even though the underlying attention computation still has approximately quadratic computational work.

Measured naive-attention quadratic coefficient from the plot cell: 9.5367431641e-07 GB/token².

### Sustained load

The final five-minute throughput was 96.61% of the peak throughput measured during the first 30 seconds. The notebook provides a conservative clock-based assessment, which should be checked against `thermal.png`, `thermal_5060.csv`, and the NVIDIA clock-event reasons.
