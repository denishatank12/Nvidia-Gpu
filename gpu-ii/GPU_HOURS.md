# GPU-hour record

GPU: Tesla T4 (`GPU-d3085845-9dcd-3867-c44d-22ef57457cde`)

The original HW4.a configuration is the accuracy-oriented recommendation from the measured runs. Its wall time including validation was 244.93 seconds: 1 GPU × 244.93 seconds ÷ 3600 = **0.0680 GPU-hours**.

The faster combined candidate had a wall time of 67.10 seconds, but lower test accuracy.

Calculation: 1 GPU × 67.10 seconds ÷ 3600 = **0.0186 GPU-hours**.

These are individual three-epoch training runs including validation, not the cost of the entire benchmark suite, downloads, or idle allocation time.
