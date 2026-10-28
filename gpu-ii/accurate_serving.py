"""Real request timing and observed vLLM KV-block occupancy, pinned to vLLM 0.10.2 V0.

Each context attempt is a fresh process. Admission failure, cache-capacity
failure, CUDA OOM, and the model's position limit are reported separately.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["VLLM_USE_V1"] = "0"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
NEW_TOKENS = 64
PROMPTS = [
    "Explain why batching can improve GPU throughput in two sentences.",
    "Describe one benefit and one cost of mixed precision training.",
    "Explain latency versus throughput for an interactive chatbot.",
    "Write a short positive movie review without naming a movie.",
    "Explain what a CUDA out-of-memory error means.",
    "Explain gradient accumulation to a student in two sentences.",
    "Why does an autoregressive model hide future tokens?",
    "Give two practical ways to reduce neural network memory use.",
]


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(path).with_name(Path(path).name + '.writing')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def memory_utilization():
    try:
        total, free = map(float, subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            text=True).splitlines()[0].split(","))
    except (ValueError, subprocess.CalledProcessError):
        # Some WSL NVML fields are unavailable; query the CUDA driver instead.
        import torch
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        total, free = total_bytes / 2**20, free_bytes / 2**20
    value = min(0.80, 0.85 * free / total)
    if value < 0.35:
        raise RuntimeError("Not enough free VRAM. Close other GPU programs before benchmarking.")
    return value


def request_timing(output, submitted_at=None):
    """Use each request's own engine timestamps, never the enclosing batch time."""
    metrics = output.metrics
    if metrics is None or metrics.finished_time is None or metrics.first_token_time is None:
        raise RuntimeError("vLLM did not expose completed request timestamps; do not substitute batch latency.")
    arrival = metrics.arrival_time if submitted_at is None else submitted_at
    latency = 1000 * (metrics.finished_time - arrival)
    first_token = 1000 * (metrics.first_token_time - arrival)
    if not (0 <= first_token <= latency and latency > 0):
        raise RuntimeError("Invalid per-request timestamp ordering.")
    return latency, first_token


def fixed_utilization():
    # Freeze one memory budget for HW4.a, HW4.5 and resumed context probes.
    path = OUT / 'serving_configuration.json'
    if path.exists():
        value = float(json.loads(path.read_text())['gpu_memory_utilization'])
    else:
        value = memory_utilization()
        write(path, {'gpu_memory_utilization': value, 'model': MODEL,
                     'note': 'Chosen once from free VRAM, then reused unchanged across serving stages and context probes.'})
    assert 0 < value <= .80
    return value


def summarize(configuration, batch_size, raw, wall, generated):
    return {"configuration": configuration, "batch_size": batch_size,
            "tokens_per_second": generated / wall,
            "p50_latency_ms": float(np.percentile([r["latency_ms"] for r in raw], 50)),
            "p95_latency_ms": float(np.percentile([r["latency_ms"] for r in raw], 95)),
            "requests_measured": len(raw), "generated_tokens": generated,
            "measured_wall_seconds": wall,
            "measured_batches": len({r['repeat'] for r in raw})}


def make_llm(utilization, context=4096, max_sequences=32, token_budget=4096):
    import vllm
    from vllm import LLM
    if vllm.__version__ != "0.10.2":
        raise RuntimeError("Use the isolated notebook environment with vLLM 0.10.2.")
    return LLM(model=MODEL, dtype="float16", seed=8834, enforce_eager=True,
               max_model_len=context, max_num_seqs=max_sequences,
               max_num_batched_tokens=token_budget, gpu_memory_utilization=utilization,
               enable_prefix_caching=False, enable_chunked_prefill=False,
               disable_async_output_proc=True, generation_config="vllm")


def benchmark(prefix, requests=100, minimum_batches=20):
    import torch
    from transformers import AutoTokenizer, pipeline
    from vllm import SamplingParams
    if not torch.cuda.is_available():
        raise RuntimeError("A Linux CUDA GPU runtime is required.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    hf = pipeline("text-generation", model=MODEL, tokenizer=tokenizer, device=0,
                  model_kwargs={"torch_dtype": torch.float16})
    hf_kwargs = dict(max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS,
                     do_sample=False, return_tensors=True)
    hf(PROMPTS[0], **hf_kwargs)
    raw_hf = []
    wall, generated = 0.0, 0
    examples = []
    for i in range(requests):
        prompt = PROMPTS[i % len(PROMPTS)]
        prompt_length = len(tokenizer.encode(prompt, add_special_tokens=True))
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = hf(prompt, **hf_kwargs)[0]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        ids = result["generated_token_ids"]
        if hasattr(ids, "tolist"): ids = ids.tolist()
        if ids and isinstance(ids[0], list): ids = ids[0]
        count = len(ids) - prompt_length
        assert count == NEW_TOKENS, (count, prompt_length)
        raw_hf.append({"configuration": "HF pipeline, batch 1", "batch_size": 1,
                       "repeat": i, "request_id": str(i), "latency_ms": seconds * 1000,
                       "first_token_ms": None, "generated_tokens": count})
        wall += seconds
        generated += count
        if (i + 1) % 10 == 0: print(f'HF measured requests: {i + 1}/{requests}', flush=True)
        if i < 2: examples.append({"configuration": "HF pipeline, batch 1", "batch_size": 1, "prompt": prompt, "answer": tokenizer.decode(ids[prompt_length:], skip_special_tokens=True)})
    rows = [summarize("HF pipeline, batch 1", 1, raw_hf, wall, generated)]
    del hf
    gc.collect()
    torch.cuda.empty_cache()
    utilization = fixed_utilization()
    llm = make_llm(utilization)
    pool_bytes = cache_details(llm)[1]
    sampling = SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, ignore_eos=True)
    raw_vllm = []
    for batch in (1, 8, 32):
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(batch)]
        # Warm each measured batch shape with the same output length.
        for _ in range(2): llm.generate(prompts, sampling, use_tqdm=False)
        local = []
        wall, generated = 0.0, 0
        for repeat in range(max(minimum_batches, math.ceil(requests / batch))):
            prompts = [PROMPTS[(repeat * batch + i) % len(PROMPTS)] for i in range(batch)]
            torch.cuda.synchronize()
            submitted_at = time.time()
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            torch.cuda.synchronize()
            wall += time.perf_counter() - start
            for output in outputs:
                latency, first_token = request_timing(output, submitted_at)
                count = len(output.outputs[0].token_ids)
                assert count == NEW_TOKENS
                generated += count
                local.append({"configuration": f"vLLM, batch {batch}", "batch_size": batch,
                              "repeat": repeat, "request_id": output.request_id,
                              "latency_ms": latency, "first_token_ms": first_token,
                              "engine_arrival_time": output.metrics.arrival_time,
                              "engine_finished_time": output.metrics.finished_time,
                              "client_submission_time": submitted_at,
                              "generated_tokens": count})
            if repeat == 0:
                examples.append({"configuration": f"vLLM, batch {batch}", "batch_size": batch, "prompt": prompts[0], "answer": outputs[0].outputs[0].text})
            if (repeat + 1) % 5 == 0: print(f'vLLM batch {batch}: {repeat + 1} measured batches', flush=True)
        rows.append(summarize(f"vLLM, batch {batch}", batch, local, wall, generated))
        raw_vllm.extend(local)
    result = {"model": MODEL, "precision": "float16", "max_model_len": 4096,
              "gpu_memory_utilization": utilization, "prefix_caching": False,
              "observed_cache_pool_bytes": pool_bytes,
              "latency_definition": "HF pipeline wall time for each batch-1 request; vLLM individual finished_time minus client submission time, including tokenization, admission, and queue time. Each request has its own completion timestamp; engine arrival_time is also saved.",
              "output_tokens": NEW_TOKENS, "rows": rows, "examples": examples}
    write(OUT / f"{prefix}serving_summary.json", result)
    pd.DataFrame(rows).to_csv(OUT / f"{prefix}serving_benchmark.csv", index=False)
    pd.DataFrame(raw_hf + raw_vllm).to_csv(OUT / f"{prefix}request_timings.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def cache_details(llm):
    """Read actual allocated cache tensor storage, not a model-config formula."""
    engine = llm.llm_engine
    worker = engine.model_executor.driver_worker.worker
    caches = worker.cache_engine
    if len(caches) != 1 or len(engine.scheduler) != 1:
        raise RuntimeError("This instrumented experiment requires one GPU and one pipeline stage.")
    cache = caches[0]
    storages = {}
    for tensor in cache.gpu_cache:
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
    pool_bytes = sum(storages.values())
    blocks = int(cache.num_gpu_blocks)
    if blocks <= 0 or pool_bytes % blocks:
        raise RuntimeError("Cannot derive the observed cache block storage size.")
    return engine.scheduler[0], pool_bytes, blocks, pool_bytes // blocks, int(cache.block_size)


def context_worker(length, destination, utilization):
    from transformers import AutoConfig, AutoTokenizer
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    config = AutoConfig.from_pretrained(MODEL)
    limit = int(config.max_position_embeddings)
    if length + 1 > limit:
        raise ValueError("Requested prompt exceeds the unmodified model's position limit.")
    # Provision one simultaneous eight-request prefill, rather than accepting
    # success when a synchronous call silently serves the eight requests serially.
    llm = make_llm(utilization, context=length + 1, max_sequences=8, token_budget=8 * length)
    scheduler, pool_bytes, blocks, bytes_per_block, block_size = cache_details(llm)
    peak = {"used_blocks": 0, "running_requests": 0, "simultaneous_eight": False}
    original_schedule = scheduler.schedule

    def observed_schedule(*args, **kwargs):
        result = original_schedule(*args, **kwargs)
        used = blocks - scheduler.block_manager.get_num_free_gpu_blocks()
        peak["used_blocks"] = max(peak["used_blocks"], used)
        running = len(scheduler.running)
        peak["running_requests"] = max(peak["running_requests"], running)
        # With chunked prefill disabled every admitted prompt is allocated in full.
        peak["simultaneous_eight"] |= running == 8 and used >= 8 * math.ceil(length / block_size)
        return result

    scheduler.schedule = observed_schedule
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    unit = tokenizer.encode(" context", add_special_tokens=False)
    assert unit
    ids = (unit * math.ceil(length / len(unit)))[:length]
    outputs = llm.generate([TokensPrompt(prompt_token_ids=list(ids)) for _ in range(8)],
                          SamplingParams(temperature=0, max_tokens=1, ignore_eos=True), use_tqdm=False)
    assert len(outputs) == 8 and all(len(o.outputs[0].token_ids) == 1 for o in outputs)
    data = {"context_length": length, "batch_size": 8,
            "status": "success" if peak['simultaneous_eight'] else "batch8_admission_limit",
            "cache_pool_bytes": pool_bytes, "cache_pool_blocks": blocks,
            "bytes_per_cache_block": bytes_per_block, "tokens_per_block": block_size,
            "peak_used_cache_blocks": peak['used_blocks'],
            "measured_active_kv_bytes": peak['used_blocks'] * bytes_per_block,
            "peak_running_requests": peak['running_requests'],
            "all_eight_resident_together": peak['simultaneous_eight'],
            "gpu_memory_utilization": utilization, "configured_max_model_len": length + 1,
            "prefill_token_budget": 8 * length,
            "method": "Actual tensor-storage bytes per cache block multiplied by peak observed allocated GPU blocks during scheduler admission; prefix caching and chunked prefill disabled."}
    write(destination, data)
    print(json.dumps(data), flush=True)


def capacity_error(text):
    lowered = text.lower()
    if "out of memory" in lowered: return "cuda_oom"
    if any(s in lowered for s in ("no available memory for the cache blocks", "no available memory for cache blocks", "maximum number of tokens that can be stored in kv cache", "larger than the maximum number of tokens that can be stored")):
        return "kv_cache_capacity"
    return None


def context_sweep():
    from transformers import AutoConfig
    limit = int(AutoConfig.from_pretrained(MODEL).max_position_embeddings)
    utilization = fixed_utilization()
    records = {}

    def attempt(length):
        path = OUT / f"context_{length}.json"
        if path.exists():
            record = json.loads(path.read_text())
        else:
            print('Starting isolated simultaneous-batch-8 context probe:', length, flush=True)
            command = [sys.executable, __file__, "context_worker", "--length", str(length),
                       "--destination", str(path), "--utilization", str(utilization)]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            transcript = completed.stdout + "\n" + completed.stderr
            (OUT / f"context_{length}_console.txt").write_text(transcript)
            if completed.returncode == 0 and path.exists():
                record = json.loads(path.read_text())
            else:
                status = capacity_error(transcript)
                if status is None:
                    raise RuntimeError(f"Context probe {length} failed for a non-capacity reason. See {OUT / f'context_{length}_console.txt'}\n{transcript[-2000:]}")
                record = {"context_length": length, "batch_size": 8, "status": status,
                          "error": transcript[-2500:], "gpu_memory_utilization": utilization}
                write(path, record)
        records[length] = record
        print("Context", length, record['status'], flush=True)
        return record['status'] == 'success'

    good, bad = 0, None
    for length in sorted({1, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, limit - 1}):
        if attempt(length): good = length
        else:
            bad = length
            break
    if good == 0:
        raise RuntimeError("An eight-request concurrent context did not fit even at length 1.")
    # Refine the successful/failing bracket without claiming an exact token boundary.
    if bad is not None:
        while bad - good > 128:
            middle = ((good + bad) // 2 // 16) * 16
            if middle <= good: break
            if attempt(middle): good = middle
            else: bad = middle
    result = {"model": MODEL, "original_position_limit": limit,
              "gpu_memory_utilization": utilization, "longest_verified_batch8_prompt": good,
              "first_failing_bracket": bad, "boundary_resolution_tokens": None if bad is None else bad - good,
              "limit_kind": "model_position_limit" if bad is None else records[bad]['status'],
              "records": [records[k] for k in sorted(records)],
              "configuration_note": "Each probe provisions a simultaneous eight-request prefill and max_model_len=prompt_length+1, in a fresh V0 engine. The resulting reserved cache pool is measured separately for each configuration. This is an engine-configuration capacity bracket, not a universal hardware-only OOM boundary."}
    write(OUT / "context_summary.json", result)
    pd.DataFrame(result['records']).to_csv(OUT / "kv_cache_context.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['benchmark', 'context', 'context_worker'])
    parser.add_argument('--prefix', default='')
    parser.add_argument('--requests', type=int, default=100)
    parser.add_argument('--minimum-batches', type=int, default=20)
    parser.add_argument('--length', type=int)
    parser.add_argument('--destination')
    parser.add_argument('--utilization', type=float)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.task == 'benchmark': benchmark(args.prefix, args.requests, args.minimum_batches)
    elif args.task == 'context': context_sweep()
    else: context_worker(args.length, args.destination, args.utilization)
