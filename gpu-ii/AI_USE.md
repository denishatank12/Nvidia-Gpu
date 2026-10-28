# AI-use appendix

## What I used the assistant for

I used an AI assistant to prepare the training and benchmarking code, organize the notebook, add smoke tests, and draft the reports. I ran the notebook in Google Colab on a Tesla T4 and supplied the executed notebook and results ZIP for checking. The implementation and initial written explanations were assistant-generated. I am not claiming that I wrote them independently.

## A specific mistake in the earlier code

The earlier benchmark used this line:

```python
latencies_ms.extend([elapsed * 1000] * batch_size)
```

It copied the same batch duration into every request's latency record. That was not a measurement of when each request finished.

## How the problem was found

The assistant found it while comparing the measurement code with the assignment requirements. The old records could not show separate request completion times because all requests in a batch were assigned the enclosing batch time. This was a measurement problem, not a model-training failure.

## What changed and what was checked

The new code records each vLLM request's own finish time relative to client submission. It saves the individual observations, uses full-length warm-ups, and calculates p50 and p95 from those records. Batch wall time is used separately to calculate total token throughput. During the audit, the assistant independently recalculated these values from the saved CSV files.

The run also includes a fair training comparison: all six controlled configurations use three epochs, the same 10,500 training examples, effective batch 16, and 657 completed optimizer updates. Maximum-batch experiments are reported separately because their update counts differ. KV-cache measurements come from actual tensor storage and observed scheduler allocations rather than a formula alone.

The assistant checked the dataset hash and exact split, reloaded the saved checkpoint, reproduced the two custom predictions and five reported classification errors, and checked the archive's integrity. Those checks did not retrain the model. The measured results come from the T4 run; the code review and report edits were additional assistant work. I remain responsible for understanding and explaining the submitted work.
