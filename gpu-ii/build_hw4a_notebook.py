#!/usr/bin/env python3
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

cells = [
    nbf.v4.new_markdown_cell(
        """# HW4.a: BERT Fine-Tuning and vLLM Inference

**Student:** Denisha Ketan Tank  
**Parameters:** SID4 = 8834, SEED = 8834, SLICE = 834, HP_ID = 2, CLS_A = 4, CLS_B = 6  
**Hardware:** one NVIDIA GeForce RTX 4090

This notebook creates the baseline required by HW4.5. It fine-tunes BERT on the fixed Twitter sentiment subset, saves and reloads the classifier, analyzes class behavior and mistakes, and compares a plain Hugging Face pipeline with vLLM at batch sizes 1, 8, and 32. The resulting `hw4a_baseline.json` is read directly by the HW4.5 report."""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
from datetime import datetime
import json
import os
import subprocess

ROOT = Path.cwd().resolve()
SETTINGS = ROOT / "settings.json"
LOG = ROOT / "RUN_LOG_HW4A.txt"

assert (ROOT / "hw4a_training.py").exists(), "Open this notebook from the gpu-ii folder."
if not (ROOT / ".venv" / "bin" / "python").exists():
    subprocess.run(["bash", "setup_rtx4090.sh"], cwd=ROOT, check=True)

PYTHON = ROOT / ".venv" / "bin" / "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

LOG.write_text(
    f"HW4.a started: {datetime.now().astimezone().isoformat()}\\n"
    "SID4=8834 SEED=8834 SLICE=834 HP_ID=2 CLS_A=4 CLS_B=6\\n",
    encoding="utf-8",
)

def run_logged(command, label):
    command = [str(item) for item in command]
    print(f"\\n[{label}] {' '.join(command)}", flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\\n[{datetime.now().astimezone().isoformat()}] {label}\\n")
        handle.write("COMMAND: " + " ".join(command) + "\\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
        handle.write(f"EXIT CODE: {return_code}\\n")
    if return_code != 0:
        raise RuntimeError(f"{label} failed. Read RUN_LOG_HW4A.txt.")

print(subprocess.check_output(["nvidia-smi", "-L"], text=True))
settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
settings"""
    ),
    nbf.v4.new_markdown_cell(
        """## 1. Data preparation

The dataset is the public Kaggle Twitter sentiment file with columns `clean_text` and `category`. The code uses at most 5,000 consecutive source rows beginning at SLICE 834, records any missing rows, and freezes one stratified 70/15/15 split with SEED 8834. The same cached tokens and split are reused by HW4.5."""
    ),
    nbf.v4.new_code_cell(
        """if not (ROOT / settings["data_path"]).exists():
    run_logged([PYTHON, "get_twitter_data.py"], "Download Twitter sentiment data")
run_logged([PYTHON, "hw45_training.py", "prepare", "--settings", SETTINGS], "Tokenize and freeze split")
manifest = json.loads((ROOT / "cache" / "dataset_manifest.json").read_text(encoding="utf-8"))
manifest"""
    ),
    nbf.v4.new_markdown_cell(
        """## 2. BERT classifier

`BertTokenizerFast` produces fixed-length tensor inputs, and `BertForSequenceClassification` is trained end to end. The script records loss and validation accuracy per epoch, test accuracy, wall time, peak allocated GPU memory, a confusion matrix, per-class precision and recall, and five mistakes from the lowest-recall class. It then reloads the saved checkpoint and runs custom inference to verify that the saved artifact works."""
    ),
    nbf.v4.new_code_cell(
        """run_logged([PYTHON, "hw4a_training.py", "--settings", SETTINGS], "BERT fine-tuning and evaluation")"""
    ),
    nbf.v4.new_markdown_cell(
        """## 3. vLLM inference benchmark

The serving experiment uses `Qwen/Qwen2.5-0.5B-Instruct`, which is different from the BERT classifier. It runs original prompts through a plain Hugging Face pipeline at batch 1 and through vLLM at batch sizes 1, 8, and 32. Throughput uses the actual generated-token count, while p50 and p95 latency use measured closed-batch wall time."""
    ),
    nbf.v4.new_code_cell(
        """run_logged(
    [PYTHON, "hw45_serving.py", "--output-prefix", "hw4a_", "--skip-context-capacity"],
    "HW4.a serving benchmark",
)"""
    ),
    nbf.v4.new_markdown_cell("## 4. Results and interpretation"),
    nbf.v4.new_code_cell(
        """run_logged([PYTHON, "finalize_hw4a.py"], "Create HW4.a tables and baseline file")
print((ROOT / "HW4A_METRICS.md").read_text(encoding="utf-8"))"""
    ),
    nbf.v4.new_code_cell(
        """import pandas as pd
from IPython.display import Image, display

display(pd.read_csv(ROOT / "outputs" / "hw4a_epoch_metrics.csv"))
display(pd.read_csv(ROOT / "outputs" / "hw4a_per_class_metrics.csv"))
display(pd.read_csv(ROOT / "outputs" / "hw4a_misclassified_examples.csv"))
display(pd.read_csv(ROOT / "outputs" / "hw4a_custom_inference.csv"))
display(pd.read_csv(ROOT / "outputs" / "hw4a_serving_benchmark.csv"))
display(Image(filename=ROOT / "outputs" / "hw4a_learning_curves.png"))
display(Image(filename=ROOT / "outputs" / "hw4a_confusion_matrix.png"))"""
    ),
    nbf.v4.new_markdown_cell("## Final automatic check"),
    nbf.v4.new_code_cell(
        """training = json.loads((ROOT / "outputs" / "hw4a_training_summary.json").read_text(encoding="utf-8"))
serving = json.loads((ROOT / "outputs" / "hw4a_serving_summary.json").read_text(encoding="utf-8"))
baseline = json.loads((ROOT / "hw4a_baseline.json").read_text(encoding="utf-8"))

required = [
    ROOT / "HW4A_METRICS.md",
    ROOT / "RUN_LOG_HW4A.txt",
    ROOT / "hw4a_baseline.json",
    ROOT / "checkpoints" / "hw4a_bert_classifier" / "config.json",
    ROOT / "outputs" / "hw4a_learning_curves.png",
    ROOT / "outputs" / "hw4a_confusion_matrix.png",
    ROOT / "outputs" / "hw4a_epoch_metrics.csv",
    ROOT / "outputs" / "hw4a_per_class_metrics.csv",
    ROOT / "outputs" / "hw4a_misclassified_examples.csv",
    ROOT / "outputs" / "hw4a_custom_inference.csv",
    ROOT / "outputs" / "hw4a_serving_benchmark.csv",
    ROOT / "outputs" / "gpu_uuid_hw4a_before.txt",
    ROOT / "outputs" / "gpu_uuid_hw4a_after.txt",
]
checks = {
    "personal parameters": training["personal_parameters"] == {"SID4": 8834, "SEED": 8834, "SLICE": 834, "HP_ID": 2, "CLS_A": 4, "CLS_B": 6},
    "RTX 4090": "4090" in training["gpu"]["name"],
    "exact row start": training["dataset_manifest"]["requested_row_range_zero_based"][0] == 834,
    "at most 5000 consecutive rows": training["dataset_manifest"]["selected_rows_before_cleaning"] <= 5000,
    "fixed 70/15/15 split": training["dataset_manifest"]["split_ratio"] == "70/15/15",
    "per-epoch values": len(training["training"]["epoch_metrics"]) == settings["epochs"],
    "confusion matrix and class metrics": len(training["training"]["per_class"]) >= 2,
    "five worst-class mistakes": len(training["training"]["five_worst_class_mistakes"]) == 5,
    "saved model reloaded": len(training["reloaded_custom_inference"]) == 2,
    "HF and vLLM batches": [row["batch_size"] for row in serving["vllm"]["metrics"]] == [1, 8, 32],
    "baseline is populated": baseline["training_wall_seconds"] is not None,
    "all evidence files": all(path.exists() for path in required),
}
for label, passed in checks.items():
    print(("PASS" if passed else "FAIL") + ": " + label)
assert all(checks.values()), "At least one HW4.a check failed."
with LOG.open("a", encoding="utf-8") as handle:
    handle.write(f"\\nCompleted: {datetime.now().astimezone().isoformat()}\\n")
    handle.write("HW4.a automatic checklist passed.\\n")
print("HW4.a is complete and its baseline is ready for HW4.5.")"""
    ),
]

nb.cells = cells
nbf.write(nb, ROOT / "Homework_4A.ipynb")
print("Created", ROOT / "Homework_4A.ipynb")
