#!/usr/bin/env python3
"""Turn measured JSON/CSV files into the required human-readable deliverables."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from hw45_serving import describe_latency_tradeoff


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


REVIEWED_DIAGNOSES = {
    1667: "The text explicitly praises Modi and his leadership, so the positive prediction fits the visible wording; the negative dataset label may be noisy, although the missing original context prevents certainty.",
    3127: "The tweet mixes criticism of a fake promise with a rhetorical comparison between political leaders; the positive prediction misses the critical tone, but the text alone does not establish which phrase drove it.",
    4326: "Words such as looting, hypocrite, and trolls convey criticism; the neutral prediction misses these negative cues in the informal wording.",
    5127: "The writer mocks the other person with appearing monkey and wasted that time; treating this as neutral loses the criticism expressed through the rhetorical question.",
    4162: "The tweet attacks Congress but ends by supporting Modi, so it contains different sentiments toward different targets; the positive output may follow the ending rather than the overall negative label.",
    2008: "The phrase crookcheap mentality is an insult, while respect for ordinary hardworking Indians is positive wording; the positive prediction may over-weight that praise and miss the insult.",
    4206: "The visible text praises a leader rising through hard work and says India stands by him; the positive prediction is reasonable from this wording, so the negative label may be noisy or depend on missing context.",
    4772: "The tweet mentions hate but also praises a welfare proposal and expresses hope and support; the mixed wording makes a single sentiment label difficult, and the positive output agrees with several explicit cues.",
    5542: "Modi era finished expresses a negative judgment in a short tweet; the phrase game changer may have supplied a competing positive cue, although this is only a possible explanation.",
}


def diagnose(text: str, source_row: Any = None) -> str:
    """Use reviewed examples, not keyword rules pretending to explain a model."""
    if source_row is not None and int(source_row) in REVIEWED_DIAGNOSES:
        return REVIEWED_DIAGNOSES[int(source_row)]
    return "This example needs individual review; the model's internal reason cannot be inferred from its predicted label alone."


def technique_table(training: dict[str, Any]) -> list[str]:
    runs = training["full_runs"]
    labels = [
        ("baseline", "HW4.5 FP32 baseline at maximum batch"),
        ("mixed_precision", "Mixed precision"),
        ("gradient_checkpointing", "Gradient checkpointing"),
        ("gradient_accumulation", "Gradient accumulation"),
        ("optimizer_8bit", "8-bit optimizer"),
        ("best_combination", "Combined candidate: AMP + 8-bit optimizer"),
    ]
    lines = [
        "| Technique | Physical / effective batch | Optimizer updates | Throughput (samples/s) | Peak memory (GB) | Test accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in labels:
        row = runs[key]
        lines.append(
            f"| {label} | {row['physical_batch']} / {row['effective_batch']} | {row['optimizer_steps']} | {fmt(row['throughput_samples_per_second'])} | "
            f"{fmt(row['peak_memory_allocated_gb'])} | {fmt(100 * row['test_accuracy'])}% |"
        )
    return lines


def ranking_table(training: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    original_batch = training["settings"]["original_batch"]
    scaling = pd.DataFrame(training["batch_scaling"])
    scaling = scaling[(scaling["status"] == "success") & (scaling["physical_batch"] == original_batch)]
    common_base = scaling[scaling["technique"] == "baseline"].iloc[0]
    rows = []
    for technique in ("mixed_precision", "gradient_checkpointing", "optimizer_8bit", "gradient_accumulation"):
        if technique == "gradient_accumulation":
            base = training["full_runs"]["baseline"]
            row = training["full_runs"][technique]
            assert base["effective_batch"] == row["effective_batch"]
            reference = f"effective batch {row['effective_batch']}, {row['epochs']} epochs"
        else:
            base = common_base
            row = scaling[scaling["technique"] == technique].iloc[0]
            reference = f"physical batch {original_batch}, {int(row['epochs'])} epoch"
        assert base["optimizer_steps"] == row["optimizer_steps"]
        assert base["samples_seen"] == row["samples_seen"]
        saved = float(base["peak_memory_allocated_gb"] - row["peak_memory_allocated_gb"])
        lost = float(base["throughput_samples_per_second"] - row["throughput_samples_per_second"])
        score = float("inf") if lost <= 0 and saved > 0 else (saved / lost if lost > 0 else float("-inf"))
        rows.append(
            {
                "technique": technique,
                "reference": reference,
                "memory_saved_gb": saved,
                "throughput_change": -lost,
                "memory_saved_per_throughput_lost": score,
            }
        )
    rows.sort(key=lambda item: item["memory_saved_per_throughput_lost"], reverse=True)
    lines = [
        "| Rank within these controls | Technique | Matched FP32 reference | Memory saved (GB) | Throughput change (samples/s) | GB saved per sample/s lost |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        score = "no throughput loss" if row["memory_saved_per_throughput_lost"] == float("inf") else ("no memory saving" if row["memory_saved_per_throughput_lost"] == float("-inf") else fmt(row["memory_saved_per_throughput_lost"], 4))
        lines.append(
            f"| {rank} | {row['technique']} | {row['reference']} | {fmt(row['memory_saved_gb'])} | "
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
        "These are three-epoch runs at different batch sizes, not a controlled test of accuracy effects. A larger batch processes the same examples with fewer optimizer updates. The table therefore describes the measured speed, memory, and quality trade-off of each complete configuration; it cannot show that a technique alone caused an accuracy change.",
        "",
        f"The original HW4.a run used batch {prior.get('batch_size', 'not recorded')} and reached {fmt(100 * prior['test_accuracy'])}% test accuracy." if prior.get("test_accuracy") is not None else "The original HW4.a accuracy is not available for comparison.",
        "",
        "### Controlled one-epoch evidence",
        "",
        "The following probes used the same physical and effective batch, examples, epoch count, and optimizer-update budget. They are separate one-epoch measurements, not the three-epoch maximum-batch runs above.",
        "",
        "| Technique | Physical batch | Optimizer updates | Throughput (samples/s) | Peak memory (GB) | Test accuracy |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {row['technique']} | {int(row['physical_batch'])} | {int(row['optimizer_steps'])} | {fmt(row['throughput_samples_per_second'])} | {fmt(row['peak_memory_allocated_gb'])} | {fmt(100 * row['test_accuracy'])}% |"
            for _, row in scaling[
                (scaling["status"] == "success")
                & (scaling["physical_batch"] == training['settings']['original_batch'])
                & scaling["technique"].isin(["baseline", "mixed_precision", "gradient_checkpointing", "optimizer_8bit"])
            ].iterrows()
        ],
        "",
        f"At the common physical batch, gradient checkpointing changed throughput by **{checkpoint_change:+.1f}%**. That slowdown is expected because checkpointed activations are recomputed during backpropagation. Gradient accumulation used physical batch {runs['gradient_accumulation']['physical_batch']} × {runs['gradient_accumulation']['accumulation_steps']} steps = effective batch {runs['gradient_accumulation']['effective_batch']}; its wall time for {runs['gradient_accumulation']['optimizer_steps']} optimizer steps was {fmt(runs['gradient_accumulation']['training_wall_seconds_including_validation'])} seconds.",
        f"For accumulation, the matched direct-batch baseline used effective batch {baseline['effective_batch']}, {baseline['optimizer_steps']} optimizer updates, and {baseline['samples_seen']} examples across {baseline['epochs']} epochs. Its wall time was {fmt(baseline['training_wall_seconds_including_validation'])} seconds, compared with {fmt(runs['gradient_accumulation']['training_wall_seconds_including_validation'])} seconds for accumulation. Accumulation lowers the physical batch without claiming identical predictions: dropout and floating-point reduction order can still differ.",
        "",
        "### Memory saved per unit of throughput lost",
        "",
        *rank_lines,
        "",
        "The ranking includes all four independent techniques and excludes the combined candidate. AMP, checkpointing, and the 8-bit optimizer use the one-epoch common-physical-batch control. Accumulation uses the three-epoch matched-effective-batch control described above. This is a descriptive ranking against the stated references, not a universal ranking across identical workloads. Positive throughput change means faster; a speedup with memory savings has no finite memory-per-throughput-loss ratio. These short single-run differences include timing noise.",
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
            describe_latency_tradeoff(serving_rows),
            "",
            f"The measured longest successful context at batch 8 was **{serving['vllm']['longest_context_batch_8']} tokens per request**. `outputs/kv_cache_context.csv` gives both the active KV-cache requirement and the GPU memory reported around each request. The active-cache calculation uses the model's actual layer, KV-head, head-dimension, and FP16 byte counts. vLLM reserves a cache pool when its engine starts, so total process memory can stay nearly flat while the amount of that pool occupied by active tokens grows.",
            f"The configured model context limit was {serving['vllm']['model_max_length']} tokens, including the generated token. "
            + ("Every tested length succeeded, so no GPU-memory exhaustion boundary was observed. The largest successful prompt is a tested lower bound on supported prompt length, not a measured memory wall. Extending past the configured model limit would test a different configuration. " if all(row['status'] == 'success' for row in serving['vllm']['context_results']) else "The context table distinguishes successful lengths from failed attempts; a failed request alone does not prove GPU-memory exhaustion unless its recorded error says so. ")
            + "The KV-cache column is calculated active storage, not an isolated measurement of live KV tensors. The measured GPU-used column also includes the reserved pool, model weights, and workspaces.",
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
            f"Compared with my recorded HW4.a training time, the HW4.5 maximum-batch baseline changed by {training_change:+.1f}%. "
            f"However, the batch changed from {prior['batch_size']} to {baseline['physical_batch']}, so this is not a hardware-only speed comparison. "
            "Both serving stages used the same recorded GPU, FP16 precision, and closed-batch latency definition. "
            "The small serving differences do not establish a hardware improvement; timing variability may contribute. "
            "The exact earlier values remain in `hw4a_baseline.json`."
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
            f"The tested combined candidate was AMP plus the 8-bit optimizer: peak allocated memory {fmt(best['peak_memory_allocated_gb'])} GB, throughput {fmt(best['throughput_samples_per_second'])} samples/s, and measured training wall time {fmt(best['training_wall_seconds_including_validation'])} seconds. This three-epoch run used **{fmt(best['training_wall_seconds_including_validation'] / 3600, 4)} GPU-hours**: one GPU × {fmt(best['training_wall_seconds_including_validation'])} seconds ÷ 3600. This is not the GPU cost of the whole benchmark suite.",
            "",
            f"The combined candidate reached {fmt(100 * best['test_accuracy'])}% test accuracy; AMP alone reached {fmt(100 * runs['mixed_precision']['test_accuracy'])}% at {fmt(runs['mixed_precision']['throughput_samples_per_second'])} samples/s. The combination used {fmt(runs['mixed_precision']['peak_memory_allocated_gb'] - best['peak_memory_allocated_gb'])} GB less peak memory than AMP alone at the same batch, but was not the fastest or most accurate configuration. The original HW4.a configuration achieved {fmt(100 * prior['test_accuracy'])}% test accuracy." if prior.get("test_accuracy") is not None else "The combined candidate is a speed/memory option, not a demonstrated accuracy winner.",
            "",
            "For accuracy in these measured runs, I would keep the original HW4.a configuration. If memory and speed matter more, I would test AMP at that same batch and optimizer-update budget before replacing it. AMP plus the 8-bit optimizer is useful when the additional measured memory saving matters, but the current maximum-batch run does not preserve the original accuracy. This recommendation uses validation for future configuration selection; the reported test results are descriptive, not a reason to tune repeatedly on the test set.",
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
            f"{clean_text} | {diagnose(clean_text, row['source_row'])} |"
        )
    lines.extend(
        [
            "",
            "The same examples, including their diagnoses, are saved in `outputs/baseline_misclassified_examples.csv`.",
            "",
            "## HW4.5 summary",
            "",
            f"Combined speed/memory candidate: **AMP + 8-bit optimizer** — {fmt(best['peak_memory_allocated_gb'])} GB peak allocated, {fmt(best['throughput_samples_per_second'])} samples/s, {fmt(best['training_wall_seconds_including_validation'])} seconds, {fmt(100 * best['test_accuracy'])}% test accuracy. It is not an overall winner because the original HW4.a configuration had higher accuracy. This is the HW4.5 summary and is not an HW5 baseline.",
        ]
    )
    (ROOT / "METRICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    mistakes_path = OUTPUT_DIR / "baseline_misclassified_examples.csv"
    if mistakes_path.exists():
        mistakes = pd.read_csv(mistakes_path)
        mistakes["diagnosis"] = mistakes.apply(lambda row: diagnose(str(row['text']), row['source_row']), axis=1)
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
