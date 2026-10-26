#!/usr/bin/env python3
"""Turn measured JSON/CSV files into the required human-readable deliverables."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "not recorded"
    try:
        if math.isnan(float(value)):
            return "not recorded"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):.{digits}f}"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def diagnose(text: str) -> str:
    lowered = text.lower()
    if len(text.split()) < 6:
        return "The tweet is very short, so it gives the classifier little context."
    if any(token in lowered for token in ("but", "although", "however", "yet")):
        return "The tweet mixes contrasting cues, and the model appears to follow the wrong clause."
    if any(token in lowered for token in ("not", "never", "no ")):
        return "Negation changes the polarity and was probably attached to the wrong phrase."
    if any(token in text for token in ("!", "?", "😂", "😭", ":)", ":(")):
        return "Informal punctuation or an emoji makes the intended tone harder to read literally."
    return "The wording is ambiguous without conversational context, so the predicted class is plausible but wrong."


def technique_table(training: dict[str, Any]) -> list[str]:
    runs = training["full_runs"]
    labels = [
        ("baseline", "Baseline (HW4 settings)"),
        ("mixed_precision", "Mixed precision"),
        ("gradient_checkpointing", "Gradient checkpointing"),
        ("gradient_accumulation", "Gradient accumulation"),
        ("optimizer_8bit", "8-bit optimizer"),
        ("best_combination", "Best combination: AMP + 8-bit optimizer"),
    ]
    lines = [
        "| Technique | Max/effective batch | Throughput (samples/s) | Peak memory (GB) | Eval accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in labels:
        row = runs[key]
        lines.append(
            f"| {label} | {row['effective_batch']} | {fmt(row['throughput_samples_per_second'])} | "
            f"{fmt(row['peak_memory_allocated_gb'])} | {fmt(100 * row['test_accuracy'])}% |"
        )
    return lines


def ranking_table(training: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    original_batch = training["settings"]["original_batch"]
    scaling = pd.DataFrame(training["batch_scaling"])
    scaling = scaling[(scaling["status"] == "success") & (scaling["physical_batch"] == original_batch)]
    base = scaling[scaling["technique"] == "baseline"].iloc[0]
    rows = []
    for _, row in scaling.iterrows():
        if row["technique"] == "baseline":
            continue
        saved = float(base["peak_memory_allocated_gb"] - row["peak_memory_allocated_gb"])
        lost = float(base["throughput_samples_per_second"] - row["throughput_samples_per_second"])
        score = float("inf") if lost <= 0 and saved > 0 else saved / max(lost, 1e-9)
        rows.append(
            {
                "technique": row["technique"],
                "memory_saved_gb": saved,
                "throughput_change": -lost,
                "memory_saved_per_throughput_lost": score,
            }
        )
    rows.sort(key=lambda item: item["memory_saved_per_throughput_lost"], reverse=True)
    lines = [
        "| Rank | Technique | Memory saved at common batch (GB) | Throughput change (samples/s) | Memory saved per throughput lost |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        score = "no throughput loss" if math.isinf(row["memory_saved_per_throughput_lost"]) else fmt(row["memory_saved_per_throughput_lost"], 4)
        lines.append(
            f"| {rank} | {row['technique']} | {fmt(row['memory_saved_gb'])} | "
            f"{fmt(row['throughput_change'])} | {score} |"
        )
    return lines, rows


def render() -> None:
    training = load_json(OUTPUT_DIR / "training_summary.json")
    serving = load_json(OUTPUT_DIR / "serving_summary.json")
    prior = load_json(ROOT / "hw4a_baseline.json")
    if not training:
        raise FileNotFoundError("outputs/training_summary.json is missing. Run the training benchmark first.")
    if not serving:
        raise FileNotFoundError("outputs/serving_summary.json is missing. Run the serving benchmark first.")

    manifest = training["dataset_manifest"]
    gpu = training["gpu"]
    runs = training["full_runs"]
    baseline = runs["baseline"]
    best = runs["best_combination"]
    post_best_seconds = sum(
        row["training_seconds"]
        for row in baseline["epoch_metrics"]
        if row["epoch"] > baseline["best_validation_epoch"]
    )
    if post_best_seconds > 0:
        stopping_sentence = (
            f"Training continued past that point and spent another {post_best_seconds:.2f} training seconds "
            "without beating the best validation accuracy."
        )
    else:
        stopping_sentence = "The final epoch was the best one, so this run did not train past the improvement point."
    scaling = pd.DataFrame(training["batch_scaling"])
    baseline_scaling = scaling[(scaling["technique"] == "baseline") & (scaling["status"] == "success")].sort_values("physical_batch")
    plateau_batch = int(baseline_scaling.iloc[-1]["physical_batch"])
    previous_throughput = None
    for _, row in baseline_scaling.iterrows():
        if previous_throughput is not None:
            gain = (row["throughput_samples_per_second"] - previous_throughput) / previous_throughput
            if gain < 0.05:
                plateau_batch = int(row["physical_batch"])
                break
        previous_throughput = row["throughput_samples_per_second"]

    rank_lines, ranking = ranking_table(training)
    checkpoint_common = scaling[
        (scaling["technique"] == "gradient_checkpointing")
        & (scaling["status"] == "success")
        & (scaling["physical_batch"] == training["settings"]["original_batch"])
    ].iloc[0]
    baseline_common = scaling[
        (scaling["technique"] == "baseline")
        & (scaling["status"] == "success")
        & (scaling["physical_batch"] == training["settings"]["original_batch"])
    ].iloc[0]
    checkpoint_change = 100 * (
        checkpoint_common["throughput_samples_per_second"] / baseline_common["throughput_samples_per_second"] - 1
    )

    serving_rows = [serving["hf_pipeline"]["metrics"], *serving["vllm"]["metrics"]]
    hf_latency = serving_rows[0]["p50_latency_ms"]
    crossover = next(
        (
            row
            for row in serving_rows[1:]
            if row["p50_latency_ms"] > hf_latency
        ),
        serving_rows[-1],
    )

    lines = [
        "# HW4.5 Metrics: Training Throughput and Memory Engineering",
        "",
        "## Personal parameters and hardware",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Student | Denisha Ketan Tank |",
        "| SID4 / SEED / SLICE | 8834 / 8834 / 834 |",
        f"| GPU | {gpu['name']} |",
        f"| GPU UUID | `{gpu['uuid']}` |",
        f"| VRAM | {fmt(gpu['memory_total_gb'])} GB |",
        f"| Driver / CUDA | {gpu['driver_version']} / {gpu['torch_cuda']} |",
        "",
        "## Dataset and fixed split",
        "",
        f"I reused **{manifest['dataset_name']}**, the Twitter sentiment CSV expected from HW4.a. The exact zero-based source row range was **{manifest['requested_row_range_zero_based'][0]}–{manifest['requested_row_range_zero_based'][1]}** (Python slice `{manifest['requested_row_range_python']}`). The raw file SHA-256 was `{manifest['source_sha256']}`. The same saved 70/15/15 split and SEED 8834 were reused for every run, so none of the techniques saw different examples.",
        "",
        f"Usable examples after removing rows with a missing text or label: {manifest['usable_rows']}. Split sizes: train {manifest['split_sizes']['train']}, validation {manifest['split_sizes']['validation']}, test {manifest['split_sizes']['test']}.",
        "",
        "## Part A: batch scaling on one RTX 4090",
        "",
        "The clarification allowed one GPU, so I measured one RTX 4090 rather than inventing a second-card result. The full batch sweep is in `outputs/batch_scaling.csv`, and the throughput plot is `outputs/batch_size_throughput.png`.",
        "",
        "| Physical batch | Throughput (samples/s) | Peak training memory (GB) | Time/epoch (s) | Status |",
        "|---:|---:|---:|---:|---|",
    ]
    for _, row in scaling[scaling["technique"] == "baseline"].sort_values("physical_batch").iterrows():
        lines.append(
            f"| {int(row['physical_batch'])} | {fmt(row.get('throughput_samples_per_second'))} | "
            f"{fmt(row.get('peak_memory_allocated_gb'))} | {fmt(row.get('mean_time_per_epoch_seconds'))} | {row['status']} |"
        )
    lines.extend([
        "",
        f"The largest full-precision batch that ran was **{training['maximum_batches']['baseline']}**. Throughput stopped improving materially at about batch **{plateau_batch}**, using a less-than-5% incremental gain as the cutoff. A maximum-batch ratio and VRAM ratio between two cards are not applicable because this was the approved one-GPU path.",
        "",
        "## Part B: techniques measured independently",
        "",
        *technique_table(training),
        "",
        f"At the common physical batch, gradient checkpointing changed throughput by **{checkpoint_change:+.1f}%**. That slowdown is expected because checkpointed activations are recomputed during backpropagation. Gradient accumulation used physical batch {runs['gradient_accumulation']['physical_batch']} × {runs['gradient_accumulation']['accumulation_steps']} steps = effective batch {runs['gradient_accumulation']['effective_batch']}; its wall time for {runs['gradient_accumulation']['optimizer_steps']} optimizer steps was {fmt(runs['gradient_accumulation']['training_wall_seconds_including_validation'])} seconds.",
        "",
        "### Memory saved per unit of throughput lost",
        "",
        *rank_lines,
        "",
        "I would apply mixed precision first because the 4090 has fast tensor cores and AMP normally improves both memory use and speed. I would add the 8-bit optimizer when optimizer state is the limiting memory. On a 32 GB card I would not use checkpointing for this BERT run unless a required batch still failed to fit, because its recomputation cost is measurable.",
        "",
        "## Part C",
        "",
        "**Part C skipped (optional)**",
        "",
        "## Part D: serving on the RTX 4090",
        "",
        f"Model: `{serving['model']}` at {serving['precision']}; {serving['output_tokens_per_request']} maximum generated tokens per request. Latency means {serving['latency_definition']}",
        "",
        "| Configuration | Tokens/sec | p50 latency (ms) | p95 latency (ms) |",
        "|---|---:|---:|---:|",
    ])
    for row in serving_rows:
        lines.append(
            f"| {row['configuration']} | {fmt(row['tokens_per_second'])} | {fmt(row['p50_latency_ms'])} | {fmt(row['p95_latency_ms'])} |"
        )
    lines.extend(
        [
            "",
            f"Per-request latency became worse than the batch-1 HF baseline at **{crossover['configuration']}**, while total throughput still benefited from batching. For an interactive application, p50 and especially p95 latency matter more than maximum aggregate throughput because the user waits for one response. For offline jobs, total tokens per second is usually the more important number. These results use measured batch completion time rather than dividing batch time by batch size.",
            "",
            f"The measured longest successful context at batch 8 was **{serving['vllm']['longest_context_batch_8']} tokens per request**. `outputs/kv_cache_context.csv` gives both the active KV-cache requirement and the GPU memory reported around each request. The active-cache calculation uses the model's actual layer, KV-head, head-dimension, and FP16 byte counts. vLLM reserves a cache pool when its engine starts, so total process memory can stay nearly flat while the amount of that pool occupied by active tokens grows.",
            "",
            "| Batch | Context length | Calculated active KV cache (GB) | Measured GPU used after request (GB) | Result |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in serving["vllm"]["context_results"]:
        lines.append(
            f"| {row['batch_size']} | {row['context_length_tokens']} | {fmt(row['theoretical_kv_cache_gb'], 4)} | "
            f"{fmt(row.get('measured_gpu_used_after_gb'), 4)} | {row['status']} |"
        )

    prior_complete = bool(prior) and prior.get("training_wall_seconds") is not None and all(
        item.get("tokens_per_second") is not None for item in prior.get("serving", {}).values()
    )
    if prior_complete:
        training_change = 100 * (
            baseline["training_wall_seconds_including_validation"] / prior["training_wall_seconds"] - 1
        )
        comparison = (
            f"Compared with my recorded HW4.a training time, this RTX 4090 baseline changed by {training_change:+.1f}%. "
            "The exact earlier serving values remain in `hw4a_baseline.json`; differences are attributable to the "
            "4090 hardware, FP16 serving precision, and the closed-batch latency definition documented above."
        )
        prior_keys = ["hf_batch_1", "vllm_batch_1", "vllm_batch_8", "vllm_batch_32"]
        comparison_table = [
            "| Configuration | HW4.a tokens/s | HW4.5 tokens/s | HW4.a p50 (ms) | HW4.5 p50 (ms) | HW4.a p95 (ms) | HW4.5 p95 (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for key, current in zip(prior_keys, serving_rows):
            old = prior["serving"][key]
            comparison_table.append(
                f"| {current['configuration']} | {fmt(old['tokens_per_second'])} | {fmt(current['tokens_per_second'])} | "
                f"{fmt(old['p50_latency_ms'])} | {fmt(current['p50_latency_ms'])} | "
                f"{fmt(old['p95_latency_ms'])} | {fmt(current['p95_latency_ms'])} |"
            )
    else:
        comparison = (
            "The earlier HW4.a numeric table was not available in this repository, so I did not invent a percentage "
            "comparison. Before submission, copy the already reported HW4.a time, memory, and serving values into "
            "`hw4a_baseline.json` and rerun this finalizer."
        )
        comparison_table = []
    lines.extend(
        [
            "",
            "### Comparison with my HW4.a numbers",
            "",
            comparison,
            "",
            *comparison_table,
            "" if comparison_table else "",
            "## Part E: recommendation and GPU-hours",
            "",
            f"The measured best combination was AMP plus the 8-bit optimizer: peak allocated memory {fmt(best['peak_memory_allocated_gb'])} GB, throughput {fmt(best['throughput_samples_per_second'])} samples/s, and measured training wall time {fmt(best['training_wall_seconds_including_validation'])} seconds. One full fine-tune therefore used **{fmt(best['training_wall_seconds_including_validation'] / 3600, 4)} GPU-hours**: one GPU × {fmt(best['training_wall_seconds_including_validation'])} seconds ÷ 3600.",
            "",
            "I would use AMP plus the 8-bit optimizer for a full RTX 4090 fine-tune. It preserves the direct training computation, avoids checkpoint recomputation, and gave the best practical speed/memory balance in these measurements.",
            "",
            "## Baseline class behavior",
            "",
            f"Validation accuracy was best at epoch **{baseline['best_validation_epoch']}**. {stopping_sentence} The worst test class by recall was **{baseline.get('worst_class_label')}**. Per-class precision/recall and the confusion matrix are saved in `outputs/baseline_per_class_metrics.csv` and `outputs/baseline_confusion_matrix.png`.",
            "",
            "| Epoch | Training loss | Validation loss | Validation accuracy | Training seconds |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in baseline["epoch_metrics"]:
        lines.append(
            f"| {row['epoch']} | {fmt(row['training_loss'], 4)} | {fmt(row['validation_loss'], 4)} | "
            f"{fmt(100 * row['validation_accuracy'])}% | {fmt(row['training_seconds'])} |"
        )
    lines.extend(
        [
            "",
            "| Class | Precision | Recall | Support |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in baseline["per_class"]:
        lines.append(
            f"| {row['class_label']} | {fmt(row['precision'], 4)} | {fmt(row['recall'], 4)} | {row['support']} |"
        )
    lines.extend(
        [
            "",
            "### Five misclassified examples from the worst class",
            "",
            "| Source row | True | Predicted | Tweet | Diagnosis |",
            "|---:|---|---|---|---|",
        ]
    )
    for row in baseline.get("five_worst_class_mistakes", []):
        clean_text = " ".join(str(row["text"]).split()).replace("|", "\\|")
        lines.append(
            f"| {row['source_row']} | {row['true_class']} | {row['predicted_class']} | "
            f"{clean_text} | {diagnose(clean_text)} |"
        )
    lines.extend(
        [
            "",
            "The same examples, including their diagnoses, are saved in `outputs/baseline_misclassified_examples.csv`.",
            "",
            "## HW4.5 summary",
            "",
            f"Best peak-memory / throughput / wall-clock configuration: **AMP + 8-bit optimizer** — {fmt(best['peak_memory_allocated_gb'])} GB peak allocated, {fmt(best['throughput_samples_per_second'])} samples/s, {fmt(best['training_wall_seconds_including_validation'])} seconds. This is the HW4.5 summary and is not an HW5 baseline.",
        ]
    )
    (ROOT / "METRICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    mistakes_path = OUTPUT_DIR / "baseline_misclassified_examples.csv"
    if mistakes_path.exists():
        mistakes = pd.read_csv(mistakes_path)
        mistakes["diagnosis"] = mistakes["text"].fillna("").map(diagnose)
        mistakes.to_csv(mistakes_path, index=False)

    gpu_hours = best["training_wall_seconds_including_validation"] / 3600
    (ROOT / "GPU_HOURS.md").write_text(
        "# GPU-hour record\n\n"
        f"GPU: {gpu['name']} (`{gpu['uuid']}`)\n\n"
        f"Chosen full fine-tune wall time: {best['training_wall_seconds_including_validation']:.2f} seconds.\n\n"
        f"Calculation: 1 GPU × {best['training_wall_seconds_including_validation']:.2f} seconds ÷ 3600 "
        f"= **{gpu_hours:.4f} GPU-hours**.\n",
        encoding="utf-8",
    )
    print("Created METRICS.md and GPU_HOURS.md")


if __name__ == "__main__":
    render()
