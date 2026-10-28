"""Shared report/smoke-test helpers. Real measurements are in accurate_serving.py."""
from __future__ import annotations
import numpy as np

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def kv_bytes_per_token(config, dtype_bytes=2):
    # This formula is checked as a unit test, NOT substituted for observed cache usage.
    layers = int(config.num_hidden_layers)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    dimension = int(getattr(config, "head_dim", config.hidden_size // heads))
    return 2 * layers * kv_heads * dimension * dtype_bytes


def describe_latency_tradeoff(rows):
    hf = rows[0]
    previous = hf["tokens_per_second"]
    crossover = None
    for row in rows[1:]:
        if row["p50_latency_ms"] > hf["p50_latency_ms"] and row["tokens_per_second"] > previous:
            crossover = row["configuration"]
            break
        previous = row["tokens_per_second"]
    first = (
        f"The first measured configuration with worse median request latency than HF while throughput increased was {crossover}. "
        if crossover else
        "No tested batch showed worse median request latency than HF while throughput was still increasing. "
    )
    return first + (
        "Batching can finish more tokens per second even when individual requests wait longer. "
        "For an interactive chatbot I would prioritize request latency, especially p95. "
        "For offline generation, total throughput is usually more important."
    )
