# AI-use appendix

I used an AI assistant to help organize the HW4.a baseline and HW4.5 benchmark code, check the assignment checklist, and make the run reproducible. I reviewed the code and kept the actual measurements tied to the RTX 4090 run instead of filling the report with estimated hardware numbers.

One specific issue in the first design was the suggestion to catch a CUDA out-of-memory error and continue testing larger or smaller batches inside the same Python process. The problematic idea was:

```text
catch CUDA OOM, call empty_cache(), and keep benchmarking in the same process
```

I caught the problem while reviewing how the memory-wall experiment would work. A failed allocation can leave the process fragmented, so later peak-memory and throughput readings may no longer be comparable. I changed the sweep so every batch-size probe starts in a fresh subprocess and writes one JSON result. That makes the first OOM a recorded result and gives the next technique a clean CUDA process.

I also verified that HW4.a and HW4.5 share one frozen token cache. The row range, file hash, label mapping, and split indices are saved so I can explain exactly what data each experiment used.
