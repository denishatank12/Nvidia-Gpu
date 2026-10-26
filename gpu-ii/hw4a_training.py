#!/usr/bin/env python3
"""Run the original HW4.a BERT fine-tune once and save its baseline evidence."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from hw45_training import (
    OUTPUT_DIR,
    Settings,
    capture_gpu,
    gpu_snapshot,
    load_cache,
    parse_settings,
    prepare_data,
    require_cuda,
    run_training,
)


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "hw4a_bert_classifier"


def save_figures(result: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in result["epoch_metrics"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.25))
    axes[0].plot(epochs, [row["training_loss"] for row in result["epoch_metrics"]], marker="o", label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in result["epoch_metrics"]], marker="o", label="validation")
    axes[0].set(title="HW4.a BERT loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, [100 * row["validation_accuracy"] for row in result["epoch_metrics"]], marker="o")
    axes[1].set(title="HW4.a validation accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hw4a_learning_curves.png", dpi=180)
    plt.close(fig)

    labels = [row["class_label"] for row in result["per_class"]]
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(result["confusion_matrix"], annot=True, fmt="d", cmap="Greys", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("HW4.a test confusion matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hw4a_confusion_matrix.png", dpi=180)
    plt.close()


def verify_saved_model(settings: Settings) -> list[dict]:
    """Reload the saved checkpoint and run two custom examples."""
    import torch
    from transformers import BertForSequenceClassification, BertTokenizerFast

    cache = load_cache()
    tokenizer = BertTokenizerFast.from_pretrained(CHECKPOINT)
    model = BertForSequenceClassification.from_pretrained(CHECKPOINT).cuda().eval()
    texts = [
        "The movie was thoughtful, funny, and much better than I expected.",
        "The story dragged for hours and the ending was disappointing.",
    ]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=settings.max_length,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1).cpu()
    rows = []
    for text, scores in zip(texts, probabilities):
        predicted_id = int(scores.argmax())
        rows.append(
            {
                "text": text,
                "predicted_class": cache["id_to_label"][predicted_id],
                "confidence": float(scores[predicted_id]),
            }
        )
    del model
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default=str(ROOT / "settings.json"))
    args = parser.parse_args()
    settings = parse_settings(args.settings)
    require_cuda()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    capture_gpu("hw4a_before")
    manifest = prepare_data(settings)
    gpu = gpu_snapshot()

    result = run_training(
        settings,
        technique="baseline",
        batch_size=settings.original_batch,
        epochs=settings.epochs,
        save_details=True,
        save_name=CHECKPOINT.name,
    )
    if len(result.get("five_worst_class_mistakes", [])) < 5:
        raise RuntimeError("The worst class did not produce five errors; inspect the split before submission.")
    reloaded_inference = verify_saved_model(settings)
    capture_gpu("hw4a_after")
    payload = {
        "personal_parameters": {"SID4": 8834, "SEED": 8834, "SLICE": 834, "HP_ID": 2, "CLS_A": 4, "CLS_B": 6},
        "settings": settings.__dict__,
        "gpu": gpu,
        "dataset_manifest": manifest,
        "training": result,
        "saved_model": str(CHECKPOINT),
        "reloaded_custom_inference": reloaded_inference,
    }
    (OUTPUT_DIR / "hw4a_training_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(result["epoch_metrics"]).to_csv(OUTPUT_DIR / "hw4a_epoch_metrics.csv", index=False)
    pd.DataFrame(result["per_class"]).to_csv(OUTPUT_DIR / "hw4a_per_class_metrics.csv", index=False)
    pd.DataFrame(result["five_worst_class_mistakes"]).to_csv(OUTPUT_DIR / "hw4a_misclassified_examples.csv", index=False)
    pd.DataFrame(reloaded_inference).to_csv(OUTPUT_DIR / "hw4a_custom_inference.csv", index=False)
    save_figures(result)
    print("HW4.a BERT baseline complete.")


if __name__ == "__main__":
    main()
