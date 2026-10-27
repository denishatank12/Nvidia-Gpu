#!/usr/bin/env python3
"""Create the HW4.a report and machine-readable baseline for HW4.5."""

from __future__ import annotations

import json
from pathlib import Path
from hw45_serving import describe_latency_tradeoff


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def diagnose(text: str) -> str:
    lowered = text.lower()
    if len(text.split()) < 6:
        return "The tweet is very short, so the model has little context."
    if any(word in lowered for word in ("but", "although", "however", "yet")):
        return "The tweet contains contrasting sentiment cues, and the model followed the wrong clause."
    if any(word in lowered for word in ("not", "never", "no ")):
        return "Negation changes the sentiment and appears to have been attached to the wrong phrase."
    if any(mark in text for mark in ("!", "?", "😂", "😭", ":)", ":(")):
        return "Informal punctuation or an emoji makes the intended tone harder to interpret literally."
    return "The wording is ambiguous without its conversation context, so the wrong prediction is understandable."


def f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    training_path = OUTPUT_DIR / "hw4a_training_summary.json"
    serving_path = OUTPUT_DIR / "hw4a_serving_summary.json"
    if not training_path.exists() or not serving_path.exists():
        raise FileNotFoundError("Run both HW4.a training and serving before finalizing.")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    serving = json.loads(serving_path.read_text(encoding="utf-8"))
    result = training["training"]
    manifest = training["dataset_manifest"]
    rows = [serving["hf_pipeline"]["metrics"], *serving["vllm"]["metrics"]]

    baseline = {
        "dataset_name": manifest["dataset_name"],
        "dataset_sha256": manifest["source_sha256"],
        "row_range_zero_based": manifest["requested_row_range_zero_based"],
        "split_ratio": manifest["split_ratio"],
        "seed": manifest["seed"],
        "model_name": training["settings"]["model_name"],
        "batch_size": result["physical_batch"],
        "epochs": result["epochs"],
        "training_wall_seconds": result["training_wall_seconds_including_validation"],
        "peak_gpu_memory_gb": result["peak_memory_allocated_gb"],
        "test_accuracy": result["test_accuracy"],
        "serving": {
            key: {
                "tokens_per_second": row["tokens_per_second"],
                "p50_latency_ms": row["p50_latency_ms"],
                "p95_latency_ms": row["p95_latency_ms"],
            }
            for key, row in zip(("hf_batch_1", "vllm_batch_1", "vllm_batch_8", "vllm_batch_32"), rows)
        },
    }
    (ROOT / "hw4a_baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    best_epoch = result["best_validation_epoch"]
    extra_seconds = sum(row["training_seconds"] for row in result["epoch_metrics"] if row["epoch"] > best_epoch)
    if extra_seconds:
        improvement = f"Validation accuracy stopped improving after epoch {best_epoch}. The remaining epochs cost {extra_seconds:.2f} training seconds without improving the best validation accuracy."
    else:
        improvement = f"The best validation accuracy occurred at the final epoch, epoch {best_epoch}, so this run did not continue past the best point."

    lines = [
        "# HW4.a: BERT Fine-Tuning and vLLM Inference",
        "",
        "Student: Denisha Ketan Tank",
        "",
        "## Reproducibility and data",
        "",
        "Personal parameters: SID4 = 8834, SEED = 8834, SLICE = 834, HP_ID = 2, CLS_A = 4, CLS_B = 6.",
        "",
        f"I used `{manifest['dataset_name']}` with SHA-256 `{manifest['source_sha256']}`. The exact zero-based source rows were {manifest['requested_row_range_zero_based'][0]} through {manifest['requested_row_range_zero_based'][1]} (Python slice `{manifest['requested_row_range_python']}`). After recording missing rows, the fixed 70/15/15 split contained {manifest['split_sizes']['train']} training, {manifest['split_sizes']['validation']} validation, and {manifest['split_sizes']['test']} test examples. Every split and model initialization used SEED 8834.",
        "",
        "## BERT fine-tuning",
        "",
        f"Model: `{training['settings']['model_name']}`. Batch size: {result['physical_batch']}. Epochs: {result['epochs']}. Peak allocated GPU memory: {f(result['peak_memory_allocated_gb'])} GB. Training wall time including validation: {f(result['training_wall_seconds_including_validation'], 2)} seconds. Test accuracy: {f(100 * result['test_accuracy'], 2)}%.",
        "",
        improvement,
        "",
        "| Epoch | Training loss | Validation loss | Validation accuracy | Training time (s) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result["epoch_metrics"]:
        lines.append(f"| {row['epoch']} | {f(row['training_loss'], 4)} | {f(row['validation_loss'], 4)} | {f(100 * row['validation_accuracy'], 2)}% | {f(row['training_seconds'], 2)} |")
    lines += ["", "### Per-class behavior", "", "| Class | Precision | Recall | Support |", "|---|---:|---:|---:|"]
    for row in result["per_class"]:
        lines.append(f"| {row['class_label']} | {f(row['precision'], 4)} | {f(row['recall'], 4)} | {row['support']} |")
    lines += [
        "",
        f"The worst class by recall was **{result['worst_class_label']}**. The confusion matrix is in `outputs/hw4a_confusion_matrix.png`.",
        "",
        "### Five mistakes from the worst class",
        "",
        "| Source row | True | Predicted | Tweet | Diagnosis |",
        "|---:|---|---|---|---|",
    ]
    for row in result["five_worst_class_mistakes"]:
        text = " ".join(str(row["text"]).split()).replace("|", "\\|")
        lines.append(f"| {row['source_row']} | {row['true_class']} | {row['predicted_class']} | {text} | {diagnose(text)} |")
    lines += [
        "",
        "The trained classifier was saved, reloaded from disk, and used on two custom sentences. The outputs are in `outputs/hw4a_custom_inference.csv`.",
        "",
        "## vLLM serving benchmark",
        "",
        f"I used the different Hugging Face model `{serving['model']}` with my own prompts. All configurations generated at most {serving['output_tokens_per_request']} tokens per request.",
        "",
        "| Configuration | Tokens/sec | p50 latency (ms) | p95 latency (ms) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['configuration']} | {f(row['tokens_per_second'], 2)} | {f(row['p50_latency_ms'], 2)} | {f(row['p95_latency_ms'], 2)} |")
    hf_latency = rows[0]["p50_latency_ms"]
    crossover = next((row for row in rows[1:] if row["p50_latency_ms"] > hf_latency), rows[-1])
    lines += [
        "",
        describe_latency_tradeoff(rows),
    ]
    (ROOT / "HW4A_METRICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Created HW4A_METRICS.md and populated hw4a_baseline.json.")


if __name__ == "__main__":
    main()
