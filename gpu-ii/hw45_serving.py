#!/usr/bin/env python3
"""HW4.5 vLLM serving benchmark for one RTX 4090.

Latency is measured from submission of a synchronous batch until all responses
in that batch are available.  Every request in that closed batch therefore has
the batch completion latency, while throughput uses the actual generated-token
count divided by the measured batch wall time.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
SEED = 8834
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_TOKENS = 64


PROMPTS = [
    "Explain why batching can improve GPU inference throughput in two sentences.",
    "Give one practical reason to use mixed precision during neural network training.",
    "Summarize the difference between latency and throughput for an interactive service.",
    "Write a short positive movie review without naming a specific movie.",
    "What causes a CUDA out-of-memory error during model training?",
    "Explain gradient accumulation to a student in one sentence.",
    "Why does causal language modeling prevent attention to future tokens?",
    "Name two ways to reduce neural network GPU memory use.",
] * 4


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run on the RTX 4090 system.")


def gpu_used_mib() -> int:
    line = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    values = [int(value.strip()) for value in line.splitlines() if value.strip().isdigit()]
    return sum(values)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def describe_latency_tradeoff(rows: list[dict[str, Any]]) -> str:
    """Describe an observed crossover only when the measured rows show it."""
    hf = rows[0]
    previous_throughput = hf["tokens_per_second"]
    crossover = None
    for row in rows[1:]:
        if row["p50_latency_ms"] > hf["p50_latency_ms"] and row["tokens_per_second"] > previous_throughput:
            crossover = row["configuration"]
            break
        previous_throughput = row["tokens_per_second"]
    first = (
        f"The first measured configuration with worse p50 latency than the HF baseline while throughput increased over the preceding configuration was {crossover}. "
        if crossover else
        "None of the tested configurations showed worse p50 latency than the HF baseline while also increasing throughput over the preceding configuration. "
    )
    return first + (
        "Batching can increase total work completed per second while individual requests wait longer. "
        "For an interactive application, p50 and especially p95 latency matter because a user waits for a response. "
        "For an offline job, aggregate tokens per second is usually more important."
    )


def measure_hf_pipeline(model_name: str) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer, pipeline

    print("Loading plain Hugging Face pipeline baseline...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {
        "task": "text-generation",
        "model": model_name,
        "tokenizer": tokenizer,
        "device": 0,
        "model_kwargs": {"torch_dtype": torch.float16},
    }
    generator = pipeline(**kwargs)
    # Untimed warm-up removes CUDA context/model warm-up from the benchmark.
    generator(PROMPTS[0], max_new_tokens=8, do_sample=False, return_full_text=False)
    latencies_ms: list[float] = []
    total_tokens = 0
    total_seconds = 0.0
    examples = []
    for prompt in PROMPTS[:10]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = generator(prompt, max_new_tokens=OUTPUT_TOKENS, do_sample=False, return_full_text=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        text = result[0]["generated_text"]
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        total_tokens += token_count
        total_seconds += elapsed
        latencies_ms.append(elapsed * 1000)
        examples.append({"prompt": prompt, "answer": text})
    row = {
        "configuration": "HF pipeline, batch 1",
        "batch_size": 1,
        "tokens_per_second": total_tokens / total_seconds,
        "p50_latency_ms": percentile(latencies_ms, 50),
        "p95_latency_ms": percentile(latencies_ms, 95),
        "requests_measured": len(latencies_ms),
        "generated_tokens": total_tokens,
    }
    del generator
    gc.collect()
    torch.cuda.empty_cache()
    return {"metrics": row, "examples": examples}


def kv_bytes_per_token(config, dtype_bytes: int = 2) -> int:
    layers = int(config.num_hidden_layers)
    attention_heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", attention_heads))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // attention_heads))
    # Key and value tensors are both stored for every layer.
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def measure_vllm(
    model_name: str,
    gpu_memory_utilization: float,
    measure_context_capacity: bool = True,
) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model_limit = int(getattr(config, "max_position_embeddings", 32768))
    max_model_len = min(model_limit, 32768)
    before_mib = gpu_used_mib()
    print(f"Loading vLLM with max_model_len={max_model_len}...", flush=True)
    llm = LLM(
        model=model_name,
        dtype="float16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=SEED,
        # Eager mode avoids a large CUDA-graph compilation step in Colab.
        # Both HW4.a and HW4.5 inherit the same setting for a fair comparison.
        enforce_eager=os.environ.get("GPU_ASSIGNMENT_EAGER", "0") == "1",
        generation_config="vllm",
    )
    after_mib = gpu_used_mib()
    sampling = SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS)
    llm.generate([PROMPTS[0]], SamplingParams(temperature=0.0, max_tokens=8), use_tqdm=False)

    rows = []
    examples = []
    repeats_by_batch = {1: 10, 8: 5, 32: 3}
    for batch_size in (1, 8, 32):
        latencies_ms: list[float] = []
        total_tokens = 0
        total_seconds = 0.0
        for repeat in range(repeats_by_batch[batch_size]):
            start_index = (repeat * batch_size) % len(PROMPTS)
            prompts = [PROMPTS[(start_index + index) % len(PROMPTS)] for index in range(batch_size)]
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            elapsed = time.perf_counter() - start
            generated = sum(len(output.outputs[0].token_ids) for output in outputs)
            total_tokens += generated
            total_seconds += elapsed
            # Offline synchronous generate returns the entire closed batch at once.
            latencies_ms.extend([elapsed * 1000] * batch_size)
            if repeat == 0:
                examples.append(
                    {
                        "batch_size": batch_size,
                        "prompt": prompts[0],
                        "answer": outputs[0].outputs[0].text,
                    }
                )
        rows.append(
            {
                "configuration": f"vLLM, batch {batch_size}",
                "batch_size": batch_size,
                "tokens_per_second": total_tokens / total_seconds,
                "p50_latency_ms": percentile(latencies_ms, 50),
                "p95_latency_ms": percentile(latencies_ms, 95),
                "requests_measured": len(latencies_ms),
                "generated_tokens": total_tokens,
            }
        )

    bytes_per_token = kv_bytes_per_token(config)
    context_rows = []
    longest_successful = None
    candidates = []
    if measure_context_capacity:
        candidates = [
            length
            for length in (512, 1024, 2048, 4096, 8192, 16384, 32768)
            if length < max_model_len
        ]
        candidates.append(max_model_len - 1)
    for context_length in sorted(set(candidates)):
        prompts = []
        base_ids = tokenizer.encode(" context", add_special_tokens=False) or [tokenizer.eos_token_id]
        for _ in range(8):
            ids = (base_ids * math.ceil(context_length / len(base_ids)))[:context_length]
            prompts.append(TokensPrompt(prompt_token_ids=ids))
        try:
            used_before_mib = gpu_used_mib()
            start = time.perf_counter()
            llm.generate(
                prompts,
                SamplingParams(temperature=0.0, max_tokens=1),
                use_tqdm=False,
            )
            elapsed = time.perf_counter() - start
            used_after_mib = gpu_used_mib()
            status = "success"
            longest_successful = context_length
            error = ""
        except (RuntimeError, ValueError) as exc:
            elapsed = float("nan")
            used_before_mib = gpu_used_mib()
            used_after_mib = used_before_mib
            status = "oom_or_capacity_error"
            error = str(exc)[:1000]
        context_rows.append(
            {
                "batch_size": 8,
                "context_length_tokens": context_length,
                "theoretical_kv_cache_gb": bytes_per_token * context_length * 8 / 2**30,
                "measured_gpu_used_before_gb": used_before_mib / 1024,
                "measured_gpu_used_after_gb": used_after_mib / 1024,
                "measured_gpu_used_change_gb": (used_after_mib - used_before_mib) / 1024,
                "status": status,
                "wall_seconds": elapsed,
                "error": error,
            }
        )
        if status != "success":
            break

    return {
        "metrics": rows,
        "examples": examples,
        "engine_gpu_memory_delta_mib": after_mib - before_mib,
        "kv_bytes_per_token_per_request": bytes_per_token,
        "context_results": context_rows,
        "longest_context_batch_8": longest_successful,
        "model_max_length": max_model_len,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional filename prefix, for example hw4a_ when recording the original baseline.",
    )
    parser.add_argument(
        "--skip-context-capacity",
        action="store_true",
        help="Skip the batch-8 long-context sweep when running the original HW4.a benchmark.",
    )
    args = parser.parse_args()
    require_cuda()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hf = measure_hf_pipeline(args.model)
    vllm = measure_vllm(
        args.model,
        args.gpu_memory_utilization,
        measure_context_capacity=not args.skip_context_capacity,
    )
    payload = {
        "model": args.model,
        "precision": "float16",
        "execution_mode": "eager" if os.environ.get("GPU_ASSIGNMENT_EAGER") == "1" else "cuda_graphs",
        "output_tokens_per_request": OUTPUT_TOKENS,
        "latency_definition": (
            "Closed-batch wall time from synchronous submission until every response in the batch is returned."
        ),
        "hf_pipeline": hf,
        "vllm": vllm,
    }
    prefix = args.output_prefix
    (OUTPUT_DIR / f"{prefix}serving_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    rows = [hf["metrics"], *vllm["metrics"]]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"{prefix}serving_benchmark.csv", index=False)
    pd.DataFrame(vllm["context_results"]).to_csv(
        OUTPUT_DIR / f"{prefix}kv_cache_context.csv", index=False
    )
    with (OUTPUT_DIR / f"{prefix}nvidia_smi_after_serving.txt").open("w", encoding="utf-8") as handle:
        subprocess.run(["nvidia-smi"], check=True, text=True, stdout=handle)
    with (OUTPUT_DIR / f"{prefix}gpu_uuid_after.txt").open("w", encoding="utf-8") as handle:
        subprocess.run(["nvidia-smi", "-L"], check=True, text=True, stdout=handle)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Longest successful context at batch 8:", vllm["longest_context_batch_8"])


if __name__ == "__main__":
    main()
