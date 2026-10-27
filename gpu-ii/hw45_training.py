#!/usr/bin/env python3
"""HW4.5 single-GPU BERT training and memory benchmark.

This file intentionally runs each memory-wall probe in a fresh process.  A CUDA
out-of-memory exception can leave a process fragmented, so process isolation is
both faster and more reliable than trying to continue in the same interpreter.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
CACHE_DIR = ROOT / "cache"
CHECKPOINT_DIR = ROOT / "checkpoints"

SID4 = 8834
SEED = 8834
SLICE = 834
HP_ID = 2
CLS_A = 4
CLS_B = 6


@dataclass
class Settings:
    data_path: str = "Twitter_Data.csv"
    text_column: str = "clean_text"
    label_column: str = "category"
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    start_row: int = SLICE
    row_limit: int = 5000
    seed: int = SEED
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    original_batch: int = 16
    max_probe_batch: int = 4096
    num_workers: int = 4


TECHNIQUES: dict[str, dict[str, Any]] = {
    "baseline": {"amp": False, "checkpointing": False, "optimizer8bit": False},
    "mixed_precision": {"amp": True, "checkpointing": False, "optimizer8bit": False},
    "gradient_checkpointing": {"amp": False, "checkpointing": True, "optimizer8bit": False},
    "optimizer_8bit": {"amp": False, "checkpointing": False, "optimizer8bit": True},
    # On a 4090 this is normally a better speed/memory trade-off than adding
    # checkpointing, which deliberately recomputes activations.
    "best_combination": {"amp": True, "checkpointing": False, "optimizer8bit": True},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this notebook on the RTX 4090 system.")


def gpu_snapshot() -> dict[str, Any]:
    import torch

    require_cuda()
    props = torch.cuda.get_device_properties(0)
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    line = subprocess.check_output(command, text=True).strip().splitlines()[0]
    name, uuid, memory_mib, driver = [part.strip() for part in line.split(",")]
    return {
        "name": name,
        "uuid": uuid,
        "memory_total_mib": int(memory_mib),
        "memory_total_gb": int(memory_mib) / 1024,
        "driver_version": driver,
        "torch_cuda": torch.version.cuda,
        "compute_capability": f"{props.major}.{props.minor}",
    }


def capture_gpu(label: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full = subprocess.check_output(["nvidia-smi"], text=True)
    listing = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    (OUTPUT_DIR / f"nvidia_smi_{label}.txt").write_text(full, encoding="utf-8")
    (OUTPUT_DIR / f"gpu_uuid_{label}.txt").write_text(listing, encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_data(settings: Settings, force: bool = False) -> dict[str, Any]:
    """Freeze the exact HW4.a-compatible row slice and split indices."""
    import torch
    from sklearn.model_selection import train_test_split
    from transformers import BertTokenizerFast

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "twitter_bert_cache.pt"
    manifest_path = CACHE_DIR / "dataset_manifest.json"
    preparation_settings = {
        key: getattr(settings, key)
        for key in ("model_name", "max_length", "start_row", "row_limit", "seed", "text_column", "label_column")
    }
    if cache_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = Path(settings.data_path).expanduser().resolve()
        if (
            expected.exists()
            and manifest.get("source_path") == str(expected)
            and manifest.get("source_sha256") == file_sha256(expected)
            and manifest.get("preparation_settings") == preparation_settings
        ):
            print("Using frozen tokenized data and split indices from", cache_path)
            return manifest

    data_path = Path(settings.data_path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Copy the exact Twitter CSV used for HW4.a here; "
            "the script will not replace it with a different dataset."
        )
    frame = pd.read_csv(data_path)
    missing = [column for column in (settings.text_column, settings.label_column) if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns {missing}. Available columns: {list(frame.columns)}")

    stop = min(settings.start_row + settings.row_limit, len(frame))
    if settings.start_row >= len(frame):
        raise ValueError(f"SLICE={settings.start_row} is beyond the dataset length {len(frame)}.")
    subset = frame.iloc[settings.start_row:stop].copy()
    subset["source_row"] = np.arange(settings.start_row, stop)
    selected_before_cleaning = len(subset)
    invalid = subset[settings.text_column].isna() | subset[settings.label_column].isna()
    excluded_rows = subset.loc[invalid, "source_row"].astype(int).tolist()
    subset = subset.loc[~invalid].copy()
    subset[settings.text_column] = subset[settings.text_column].astype(str)

    raw_labels = sorted(subset[settings.label_column].unique().tolist())
    label_to_id = {str(label): index for index, label in enumerate(raw_labels)}
    subset["label_id"] = subset[settings.label_column].map(lambda value: label_to_id[str(value)]).astype(int)
    positions = np.arange(len(subset))
    labels = subset["label_id"].to_numpy()
    train_positions, remainder = train_test_split(
        positions, test_size=0.30, random_state=settings.seed, stratify=labels
    )
    val_positions, test_positions = train_test_split(
        remainder,
        test_size=0.50,
        random_state=settings.seed,
        stratify=labels[remainder],
    )

    tokenizer = BertTokenizerFast.from_pretrained(settings.model_name)
    encoded = tokenizer(
        subset[settings.text_column].tolist(),
        padding="max_length",
        truncation=True,
        max_length=settings.max_length,
        return_tensors="pt",
    )
    payload = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
        "texts": subset[settings.text_column].tolist(),
        "source_rows": subset["source_row"].astype(int).tolist(),
        "splits": {
            "train": torch.tensor(train_positions, dtype=torch.long),
            "validation": torch.tensor(val_positions, dtype=torch.long),
            "test": torch.tensor(test_positions, dtype=torch.long),
        },
        "label_to_id": label_to_id,
        "id_to_label": {index: label for label, index in label_to_id.items()},
    }
    torch.save(payload, cache_path)
    manifest = {
        "dataset_name": data_path.name,
        "preparation_settings": preparation_settings,
        "source_path": str(data_path),
        "source_sha256": file_sha256(data_path),
        "raw_dataset_rows": len(frame),
        "requested_row_range_zero_based": [settings.start_row, stop - 1],
        "requested_row_range_python": f"[{settings.start_row}:{stop})",
        "selected_rows_before_cleaning": selected_before_cleaning,
        "excluded_source_rows_with_missing_text_or_label": excluded_rows,
        "usable_rows": len(subset),
        "split_sizes": {
            "train": len(train_positions),
            "validation": len(val_positions),
            "test": len(test_positions),
        },
        "split_ratio": "70/15/15",
        "seed": settings.seed,
        "model_name": settings.model_name,
        "max_length": settings.max_length,
        "text_column": settings.text_column,
        "label_column": settings.label_column,
        "label_to_id": label_to_id,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def load_cache() -> dict[str, Any]:
    import torch

    path = CACHE_DIR / "twitter_bert_cache.pt"
    if not path.exists():
        raise FileNotFoundError("Run the prepare command before benchmarking.")
    return torch.load(path, map_location="cpu", weights_only=False)


def make_loader(cache: dict[str, Any], split: str, batch_size: int, settings: Settings, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    indices = cache["splits"][split]
    dataset = TensorDataset(
        cache["input_ids"].index_select(0, indices),
        cache["attention_mask"].index_select(0, indices),
        cache["labels"].index_select(0, indices),
    )
    generator = torch.Generator().manual_seed(settings.seed)
    workers = min(settings.num_workers, os.cpu_count() or 1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        drop_last=False,
    )


def build_optimizer(model, settings: Settings, optimizer8bit: bool):
    if optimizer8bit:
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(
            model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
        )
    import torch

    kwargs = dict(lr=settings.learning_rate, weight_decay=settings.weight_decay)
    try:
        return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(model.parameters(), **kwargs)


def evaluate(model, loader) -> tuple[float, float, list[int], list[int]]:
    import torch

    model.eval()
    total_loss = 0.0
    total = 0
    predictions: list[int] = []
    targets: list[int] = []
    with torch.inference_mode():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.cuda(non_blocking=True)
            attention_mask = attention_mask.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += output.loss.item() * labels.size(0)
            batch_predictions = output.logits.argmax(dim=-1)
            predictions.extend(batch_predictions.cpu().tolist())
            targets.extend(labels.cpu().tolist())
            total += labels.size(0)
    accuracy = float(np.mean(np.asarray(predictions) == np.asarray(targets)))
    return total_loss / total, accuracy, predictions, targets


def run_training(
    settings: Settings,
    technique: str,
    batch_size: int,
    epochs: int,
    accumulation_steps: int = 1,
    save_details: bool = False,
    save_name: str | None = None,
) -> dict[str, Any]:
    import torch
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    from transformers import BertForSequenceClassification

    require_cuda()
    flags = TECHNIQUES[technique]
    cache = load_cache()
    num_labels = len(cache["label_to_id"])
    set_seed(settings.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    model = BertForSequenceClassification.from_pretrained(
        settings.model_name,
        num_labels=num_labels,
        id2label=cache["id_to_label"],
        label2id=cache["label_to_id"],
    ).cuda()
    if flags["checkpointing"]:
        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    optimizer = build_optimizer(model, settings, flags["optimizer8bit"])
    scaler = torch.amp.GradScaler("cuda", enabled=flags["amp"])
    train_loader = make_loader(cache, "train", batch_size, settings, shuffle=True)
    validation_loader = make_loader(cache, "validation", max(batch_size, 64), settings, shuffle=False)
    test_loader = make_loader(cache, "test", max(batch_size, 64), settings, shuffle=False)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    epoch_rows: list[dict[str, Any]] = []
    optimizer_steps = 0
    samples_seen = 0
    training_peak_allocated = 0
    training_peak_reserved = 0
    wall_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_samples = 0
        # Reset here so validation batches do not inflate the reported training
        # peak, especially for gradient accumulation with a small physical batch.
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        for step, (input_ids, attention_mask, labels) in enumerate(train_loader, start=1):
            input_ids = input_ids.cuda(non_blocking=True)
            attention_mask = attention_mask.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=flags["amp"]):
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                raw_loss = output.loss
                # Weight each microbatch by its share of the actual update group,
                # including a short final group at the end of an epoch.
                group_start = ((step - 1) // accumulation_steps) * accumulation_steps * batch_size
                group_samples = min(batch_size * accumulation_steps, len(train_loader.dataset) - group_start)
                loss = raw_loss * (labels.size(0) / group_samples)
            scaler.scale(loss).backward()
            should_step = step % accumulation_steps == 0 or step == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            epoch_loss += raw_loss.detach().float().item() * labels.size(0)
            epoch_samples += labels.size(0)
            samples_seen += labels.size(0)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - epoch_start
        training_peak_allocated = max(training_peak_allocated, torch.cuda.max_memory_allocated())
        training_peak_reserved = max(training_peak_reserved, torch.cuda.max_memory_reserved())
        validation_loss, validation_accuracy, _, _ = evaluate(model, validation_loader)
        row = {
            "epoch": epoch,
            "training_loss": epoch_loss / epoch_samples,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "training_seconds": train_seconds,
            "throughput_samples_per_second": epoch_samples / train_seconds,
        }
        epoch_rows.append(row)
        print(json.dumps(row), flush=True)

    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    peak_allocated = training_peak_allocated / 2**30
    peak_reserved = training_peak_reserved / 2**30
    test_loss, test_accuracy, predictions, targets = evaluate(model, test_loader)
    best_epoch = max(epoch_rows, key=lambda row: row["validation_accuracy"])["epoch"]

    precision, recall, _, support = precision_recall_fscore_support(
        targets, predictions, labels=list(range(num_labels)), zero_division=0
    )
    per_class = [
        {
            "class_id": index,
            "class_label": cache["id_to_label"][index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "support": int(support[index]),
        }
        for index in range(num_labels)
    ]
    result = {
        "status": "success",
        "technique": technique,
        "flags": flags,
        "physical_batch": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch": batch_size * accumulation_steps,
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "samples_seen": samples_seen,
        "throughput_samples_per_second": float(
            sum(row["throughput_samples_per_second"] for row in epoch_rows) / len(epoch_rows)
        ),
        "mean_time_per_epoch_seconds": float(
            sum(row["training_seconds"] for row in epoch_rows) / len(epoch_rows)
        ),
        "training_wall_seconds_including_validation": wall_seconds,
        "peak_memory_allocated_gb": peak_allocated,
        "peak_memory_reserved_gb": peak_reserved,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "best_validation_epoch": best_epoch,
        "epoch_metrics": epoch_rows,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(num_labels))).tolist(),
    }

    if save_details:
        if save_name:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            destination = CHECKPOINT_DIR / save_name
            model.save_pretrained(destination)
            from transformers import BertTokenizerFast

            BertTokenizerFast.from_pretrained(settings.model_name).save_pretrained(destination)
        test_indices = cache["splits"]["test"].tolist()
        test_texts = [cache["texts"][position] for position in test_indices]
        test_rows = [cache["source_rows"][position] for position in test_indices]
        worst_class = min(per_class, key=lambda row: row["recall"])["class_id"]
        mistakes = []
        for text, source_row, truth, predicted in zip(test_texts, test_rows, targets, predictions):
            if truth == worst_class and predicted != truth:
                mistakes.append(
                    {
                        "source_row": source_row,
                        "text": text,
                        "true_class": cache["id_to_label"][truth],
                        "predicted_class": cache["id_to_label"][predicted],
                    }
                )
            if len(mistakes) == 5:
                break
        result["worst_class_id"] = worst_class
        result["worst_class_label"] = cache["id_to_label"][worst_class]
        result["five_worst_class_mistakes"] = mistakes

    del model, optimizer, train_loader, validation_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


def probe_subprocess(settings_path: Path, technique: str, batch_size: int, result_path: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "probe",
        "--settings",
        str(settings_path),
        "--technique",
        technique,
        "--batch-size",
        str(batch_size),
        "--result",
        str(result_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = {"status": "error", "error": completed.stderr[-4000:]}
    result["stdout_tail"] = completed.stdout[-2000:]
    result["stderr_tail"] = completed.stderr[-2000:]
    return result


def run_sweep(settings: Settings, settings_path: Path, technique: str) -> list[dict[str, Any]]:
    rows = []
    batch_size = settings.original_batch
    while batch_size <= settings.max_probe_batch:
        result_path = OUTPUT_DIR / f"probe_{technique}_{batch_size}.json"
        print(f"\nProbing {technique} batch={batch_size}", flush=True)
        if os.environ.get("GPU_ASSIGNMENT_RESUME") == "1" and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") not in ("success", "oom"):
                result = probe_subprocess(settings_path, technique, batch_size, result_path)
            else:
                print("Reusing completed probe from this run.", flush=True)
        else:
            if result_path.exists():
                result_path.unlink()
            result = probe_subprocess(settings_path, technique, batch_size, result_path)
        rows.append(result)
        print(result.get("status"), result.get("error", ""), flush=True)
        if result.get("status") not in ("success", "oom"):
            raise RuntimeError(
                f"Probe failed for {technique}, batch {batch_size}: {result.get('error', result)}"
            )
        if result.get("status") != "success":
            break
        batch_size *= 2
    if rows and rows[-1].get("status") == "success":
        raise RuntimeError(
            f"Memory wall was not reached for {technique}; increase max_probe_batch above "
            f"{settings.max_probe_batch}."
        )
    return rows


def save_training_figures(summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scaling = pd.DataFrame(summary["batch_scaling"])
    successful = scaling[scaling["status"] == "success"]
    plt.figure(figsize=(8, 5))
    for technique, group in successful.groupby("technique"):
        plt.plot(group["physical_batch"], group["throughput_samples_per_second"], marker="o", label=technique)
    plt.xscale("log", base=2)
    plt.xlabel("Physical batch size")
    plt.ylabel("Training throughput (samples/s)")
    plt.title(f"{summary['gpu']['name']} BERT throughput to the memory wall")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "batch_size_throughput.png", dpi=180)
    plt.close()

    baseline = summary["full_runs"]["baseline"]
    matrix = np.asarray(baseline["confusion_matrix"])
    labels = [row["class_label"] for row in baseline["per_class"]]
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Greys", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Part A baseline confusion matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_confusion_matrix.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for technique, result in summary["full_runs"].items():
        epochs = [row["epoch"] for row in result["epoch_metrics"]]
        losses = [row["validation_loss"] for row in result["epoch_metrics"]]
        plt.plot(epochs, losses, marker="o", label=technique)
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Validation loss by memory technique")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_loss_by_technique.png", dpi=180)
    plt.close()

    epochs = [row["epoch"] for row in baseline["epoch_metrics"]]
    plt.figure(figsize=(7, 4.5))
    plt.plot(
        epochs,
        [row["training_loss"] for row in baseline["epoch_metrics"]],
        marker="o",
        label="Training loss",
    )
    plt.plot(
        epochs,
        [row["validation_loss"] for row in baseline["epoch_metrics"]],
        marker="o",
        label="Validation loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("HW4.a-compatible BERT baseline learning curves")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_training_validation_loss.png", dpi=180)
    plt.close()


def write_training_tables(summary: dict[str, Any]) -> None:
    scaling = pd.DataFrame(summary["batch_scaling"])
    scaling.to_csv(OUTPUT_DIR / "batch_scaling.csv", index=False)
    technique_rows = []
    for name, result in summary["full_runs"].items():
        technique_rows.append(
            {
                "technique": name,
                "max_batch": result["effective_batch"],
                "physical_batch": result["physical_batch"],
                "accumulation_steps": result["accumulation_steps"],
                "throughput_samples_per_second": result["throughput_samples_per_second"],
                "peak_memory_allocated_gb": result["peak_memory_allocated_gb"],
                "mean_time_per_epoch_seconds": result["mean_time_per_epoch_seconds"],
                "training_wall_seconds": result["training_wall_seconds_including_validation"],
                "test_accuracy": result["test_accuracy"],
            }
        )
    pd.DataFrame(technique_rows).to_csv(OUTPUT_DIR / "technique_comparison.csv", index=False)
    baseline = summary["full_runs"]["baseline"]
    pd.DataFrame(baseline["per_class"]).to_csv(OUTPUT_DIR / "baseline_per_class_metrics.csv", index=False)
    pd.DataFrame(baseline.get("five_worst_class_mistakes", [])).to_csv(
        OUTPUT_DIR / "baseline_misclassified_examples.csv", index=False
    )


def orchestrate_training(settings: Settings, settings_path: Path) -> dict[str, Any]:
    require_cuda()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    capture_gpu("before")
    manifest = prepare_data(settings)
    gpu = gpu_snapshot()

    all_probe_rows: list[dict[str, Any]] = []
    maximum_batches: dict[str, int] = {}
    for technique in TECHNIQUES:
        rows = run_sweep(settings, settings_path, technique)
        for row in rows:
            row.setdefault("technique", technique)
            row.setdefault("physical_batch", row.get("batch_size"))
        all_probe_rows.extend(rows)
        successes = [row for row in rows if row.get("status") == "success"]
        maximum_batches[technique] = max(int(row["physical_batch"]) for row in successes)

    full_runs: dict[str, Any] = {}
    for technique, batch_size in maximum_batches.items():
        print(f"\nTraining final {technique} run at max batch {batch_size}", flush=True)
        result_path = OUTPUT_DIR / f"full_result_{technique}.json"
        if os.environ.get("GPU_ASSIGNMENT_RESUME") == "1" and result_path.exists():
            full_runs[technique] = json.loads(result_path.read_text(encoding="utf-8"))
            print("Reusing completed full run.", flush=True)
        else:
            full_runs[technique] = run_training(
                settings,
                technique,
                batch_size,
                settings.epochs,
                save_details=technique == "baseline",
            )
            result_path.write_text(json.dumps(full_runs[technique], indent=2), encoding="utf-8")

    # Gradient accumulation is measured separately: a small physical batch
    # reaches the baseline maximum effective batch for the same epoch count.
    baseline_max = maximum_batches["baseline"]
    accumulation_steps = max(2, math.ceil(baseline_max / settings.original_batch))
    print(
        f"\nTraining gradient accumulation: physical={settings.original_batch}, "
        f"steps={accumulation_steps}, effective={settings.original_batch * accumulation_steps}",
        flush=True,
    )
    accumulation_path = OUTPUT_DIR / "full_result_gradient_accumulation.json"
    if os.environ.get("GPU_ASSIGNMENT_RESUME") == "1" and accumulation_path.exists():
        full_runs["gradient_accumulation"] = json.loads(accumulation_path.read_text(encoding="utf-8"))
    else:
        full_runs["gradient_accumulation"] = run_training(
            settings,
            "baseline",
            settings.original_batch,
            settings.epochs,
            accumulation_steps=accumulation_steps,
            save_details=False,
        )
        accumulation_path.write_text(json.dumps(full_runs["gradient_accumulation"], indent=2), encoding="utf-8")
    full_runs["gradient_accumulation"]["technique"] = "gradient_accumulation"

    capture_gpu("after_training")
    summary = {
        "personal_parameters": {
            "SID4": SID4,
            "SEED": SEED,
            "SLICE": SLICE,
            "HP_ID": HP_ID,
            "CLS_A": CLS_A,
            "CLS_B": CLS_B,
        },
        "settings": asdict(settings),
        "gpu": gpu,
        "dataset_manifest": manifest,
        "part_c": "Part C skipped (optional)",
        "batch_scaling": all_probe_rows,
        "maximum_batches": maximum_batches,
        "full_runs": full_runs,
    }
    (OUTPUT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_training_tables(summary)
    save_training_figures(summary)
    print("Training benchmark complete.")
    return summary


def parse_settings(path: str) -> Settings:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Settings(**payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--settings", required=True)
    prepare_parser.add_argument("--force", action="store_true")

    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--settings", required=True)
    probe_parser.add_argument("--technique", choices=TECHNIQUES, required=True)
    probe_parser.add_argument("--batch-size", type=int, required=True)
    probe_parser.add_argument("--result", required=True)

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--settings", required=True)

    args = parser.parse_args()
    settings = parse_settings(args.settings)
    if args.command == "prepare":
        prepare_data(settings, force=args.force)
    elif args.command == "probe":
        result_path = Path(args.result)
        try:
            result = run_training(settings, args.technique, args.batch_size, epochs=1)
        except RuntimeError as error:
            text = str(error)
            status = "oom" if "out of memory" in text.lower() else "error"
            result = {
                "status": status,
                "technique": args.technique,
                "physical_batch": args.batch_size,
                "error": text,
            }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
    elif args.command == "all":
        orchestrate_training(settings, Path(args.settings).resolve())


if __name__ == "__main__":
    main()
