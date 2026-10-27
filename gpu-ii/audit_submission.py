#!/usr/bin/env python3
"""Check recorded artifacts without rerunning or claiming to rerun GPU training."""
from __future__ import annotations

import ast
import base64
import csv
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def run_audit():
    training = read_json("outputs/training_summary.json")
    original = read_json("outputs/hw4a_training_summary.json")
    serving = read_json("outputs/serving_summary.json")
    first_serving = read_json("outputs/hw4a_serving_summary.json")
    metrics = (ROOT / "METRICS.md").read_text()
    checks = {}

    def check(label, condition):
        checks[label] = bool(condition)

    expected = {"SID4": 8834, "SEED": 8834, "SLICE": 834, "HP_ID": 2, "CLS_A": 4, "CLS_B": 6}
    check("personal parameters", training["personal_parameters"] == original["personal_parameters"] == expected)
    a, b = training["dataset_manifest"], original["dataset_manifest"]
    check("same source and preparation", a["source_sha256"] == b["source_sha256"] and a["preparation_settings"] == b["preparation_settings"])
    check("exact row range", a["requested_row_range_zero_based"] == b["requested_row_range_zero_based"] == [834, 5833])
    check("fixed split sizes", a["split_sizes"] == b["split_sizes"] == {"train": 3500, "validation": 750, "test": 750})
    check("recorded settings reproduced", read_json("settings.json") == training["settings"] == original["settings"])
    gpu = training["gpu"]
    check("actual GPU consistent", gpu == original["gpu"] and gpu["name"] == "Tesla T4")
    for name in ("gpu_uuid_before.txt", "gpu_uuid_after.txt", "gpu_uuid_hw4a_before.txt", "gpu_uuid_hw4a_after.txt"):
        check("GPU capture " + name, gpu["uuid"] in (ROOT / "outputs" / name).read_text())

    n = read_json("GPU_II_Colab_Run_All.ipynb")
    cells = [c for c in n["cells"] if c["cell_type"] == "code"]
    check("all original cells executed", [c.get("execution_count") for c in cells] == list(range(1, 10)))
    check("no uncaught original notebook error", not any(o["output_type"] == "error" for c in cells for o in c.get("outputs", [])))
    original_result = original["training"]
    check("three original epochs", len(original_result["epoch_metrics"]) == 3)
    matrix = original_result["confusion_matrix"]
    correct, total = sum(matrix[i][i] for i in range(3)), sum(map(sum, matrix))
    check("test accuracy matches confusion matrix", total == 750 and math.isclose(correct / total, original_result["test_accuracy"]))
    for i, row in enumerate(original_result["per_class"]):
        check("class metrics " + str(i), math.isclose(row["recall"], matrix[i][i] / sum(matrix[i])) and math.isclose(row["precision"], matrix[i][i] / sum(r[i] for r in matrix)))
    check("two saved-model predictions", [r["predicted_class"] for r in original["reloaded_custom_inference"]] == ["1.0", "-1.0"])

    for technique, maximum in training["maximum_batches"].items():
        rows = [r for r in training["batch_scaling"] if r["technique"] == technique]
        successful = [r["physical_batch"] for r in rows if r["status"] == "success"]
        check("maximum and next OOM " + technique, max(successful) == maximum and any(r["status"] == "oom" and r["physical_batch"] == 2 * maximum for r in rows))
    runs = training["full_runs"]
    check("all six completed training configurations", set(runs) == {"baseline", "mixed_precision", "gradient_checkpointing", "gradient_accumulation", "optimizer_8bit", "best_combination"} and all(r["status"] == "success" for r in runs.values()))
    check("matched accumulation budget", all(runs["baseline"][key] == runs["gradient_accumulation"][key] for key in ("effective_batch", "optimizer_steps", "samples_seen", "epochs")))

    from finalize_hw45 import ranking_table
    _, ranking = ranking_table(training)
    check("four independent techniques ranked", {r["technique"] for r in ranking} == {"mixed_precision", "gradient_checkpointing", "gradient_accumulation", "optimizer_8bit"})
    for prefix in ("baseline", "hw4a"):
        with (ROOT / "outputs" / (prefix + "_misclassified_examples.csv")).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        check("individual diagnoses " + prefix, len(rows) == 5 and len({r["diagnosis"] for r in rows}) == 5 and all(r["true_class"] != r["predicted_class"] for r in rows))

    for label, result in (("HW4.a", first_serving), ("HW4.5", serving)):
        check("serving batches " + label, [r["batch_size"] for r in result["vllm"]["metrics"]] == [1, 8, 32] and result["hf_pipeline"]["metrics"]["batch_size"] == 1)
        check("positive serving measurements " + label, all(0 < r["tokens_per_second"] and 0 < r["p50_latency_ms"] <= r["p95_latency_ms"] for r in [result["hf_pipeline"]["metrics"], *result["vllm"]["metrics"]]))
    context = serving["vllm"]["context_results"]
    check("batch-eight context evidence", len(context) == 7 and all(r["batch_size"] == 8 and r["status"] == "success" for r in context))
    check("context limit described honestly", serving["vllm"]["longest_context_batch_8"] == 32767 and serving["vllm"]["model_max_length"] == 32768 and "no GPU-memory exhaustion boundary was observed" in metrics)
    check("accuracy confounding disclosed", "not a controlled test of accuracy effects" in metrics and "83.60%" in metrics and "60.67%" in metrics)
    check("Part C optional skip", "Part C skipped (optional)" in metrics)

    for path in sorted((ROOT / "outputs").glob("*.png")):
        check("PNG artifact " + path.name, path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
    for name in ("RUN_LOG.txt", "RUN_LOG_HW4A.txt"):
        text = (ROOT / name).read_text()
        check("real timestamped log " + name, len(text) > 30000 and "2026-10-27" in text and "not yet been executed" not in text)
    for name in ("AI_USE.md", "METRICS.md", "HW4A_METRICS.md", "GPU_HOURS.md", "RESERVATION_RECORD.md", "MODEL_CHECKPOINT.md", "SUBMISSION_AUDIT.md"):
        check("standing artifact " + name, (ROOT / name).stat().st_size > 200)
    checkpoint_text = (ROOT / "MODEL_CHECKPOINT.md").read_text()
    check("checkpoint download documented", "releases/download/hw4-5/GPU_II_RESULTS.zip" in checkpoint_text and "pending upload" not in checkpoint_text)
    check("corrected hardware interpretation", "4090 has fast tensor cores" not in metrics and "GPU path: one NVIDIA GeForce RTX 4090" not in (ROOT / "README.md").read_text())
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(), filename=path.name)
    check("Python sources parse", True)
    check("exactly one assignment notebook", [p.name for p in ROOT.glob("*.ipynb")] == ["GPU_II_Colab_Run_All.ipynb"])
    post_run = "\n".join("".join(c['source']) for c in n['cells'] if c['cell_type'] == 'markdown' and c.get('metadata', {}).get('post_run_review'))
    check("corrected findings included in executed notebook", all((ROOT / name).read_text().strip() in post_run for name in ("HW4A_METRICS.md", "METRICS.md")))
    return {"audit_date": "2026-10-27", "scope": "Offline artifact and arithmetic audit, not a GPU rerun or a repository-access check", "checks": checks, "all_artifact_checks_passed": all(checks.values()), "limitations": ["Repository reported public at audit time; private visibility and grader access need confirmation.", "No GPU-memory exhaustion observed within the supported context range.", "Latency is closed-batch completion time, not independent request completion.", "Maximum-batch runs have different optimizer-update budgets."]}


if __name__ == "__main__":
    result = run_audit()
    for label, passed in result["checks"].items():
        print(("PASS: " if passed else "FAIL: ") + label)
    (ROOT / "outputs" / "submission_artifact_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Artifact checks passed:", result["all_artifact_checks_passed"])
    print("See SUBMISSION_AUDIT.md for limitations and external requirements.")
    raise SystemExit(0 if result["all_artifact_checks_passed"] else 1)
