#!/usr/bin/env python3
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

cells = []
cells.append(
    nbf.v4.new_markdown_cell(
        """# HW4.5: Training Throughput and Memory Engineering

**Student:** Denisha Ketan Tank  
**Hardware path:** one NVIDIA GeForce RTX 4090  
**Parameters:** SID4 = 8834, SEED = 8834, SLICE = 834, HP_ID = 2, CLS_A = 4, CLS_B = 6

This notebook keeps the HW4.a BERT/Twitter setup fixed and changes one memory technique at a time. It finds the batch-size memory wall, compares AMP, gradient checkpointing, gradient accumulation, and an 8-bit optimizer, then measures vLLM serving and KV-cache context capacity. Part C is optional and is skipped.

The HW4.a notebook creates `Twitter_Data.csv`, the frozen token cache, and `hw4a_baseline.json`. Run `Homework_4A.ipynb` first, then run this notebook so every HW4.5 comparison uses the exact same rows and split."""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """from pathlib import Path
from datetime import datetime
import json
import os
import shutil
import subprocess
import sys

ROOT = Path.cwd().resolve()
SETTINGS = ROOT / "settings.json"
LOG = ROOT / "RUN_LOG.txt"

assert (ROOT / "hw45_training.py").exists(), "Open the notebook from inside the gpu-ii folder."
if not (ROOT / ".venv" / "bin" / "python").exists():
    print("Creating the isolated CUDA environment once. This can take several minutes.")
    subprocess.run(["bash", "setup_rtx4090.sh"], cwd=ROOT, check=True)

PYTHON = ROOT / ".venv" / "bin" / "python"
assert PYTHON.exists()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

LOG.write_text(
    f"HW4.5 RTX 4090 run started: {datetime.now().astimezone().isoformat()}\\n"
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
        raise RuntimeError(f"{label} failed with exit code {return_code}. Read RUN_LOG.txt.")

print(subprocess.check_output(["nvidia-smi", "-L"], text=True))
settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
print(json.dumps(settings, indent=2))"""
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        """## Freeze the HW4.a data and split

This step takes at most 5,000 consecutive raw rows beginning at SLICE 834. Missing text or label rows are recorded rather than hidden. Tokenization happens once, and the exact 70/15/15 indices are reused by every later run."""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """assert (ROOT / settings["data_path"]).exists(), "Run Homework_4A.ipynb first to create the shared dataset."
baseline = json.loads((ROOT / "hw4a_baseline.json").read_text(encoding="utf-8"))
assert baseline["training_wall_seconds"] is not None, "Run Homework_4A.ipynb first to record the baseline."
run_logged([PYTHON, "hw45_training.py", "prepare", "--settings", SETTINGS], "Prepare fixed Twitter split")
manifest = json.loads((ROOT / "cache" / "dataset_manifest.json").read_text(encoding="utf-8"))
manifest"""
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        """## Parts A and B: training benchmark

The script scales batch size by powers of two until the first OOM. Each probe starts a new Python process, which avoids memory fragmentation after a failed allocation. The four techniques are measured independently, followed by AMP plus the 8-bit optimizer as the best practical combination for this 4090."""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """run_logged([PYTHON, "hw45_training.py", "all", "--settings", SETTINGS], "Parts A and B")"""
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        """## Part C

**Part C skipped (optional)**

The instructor clarification allows a one-GPU submission, so this notebook does not fabricate distributed measurements."""
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        """## Part D: vLLM and KV-cache benchmark

The same small instruction model is used for the plain Hugging Face pipeline and vLLM. Batch sizes 1, 8, and 32 generate the same maximum number of tokens. The capacity test uses batch 8 and generates only one new token so it reaches long contexts quickly."""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """run_logged([PYTHON, "hw45_serving.py"], "Part D serving")"""
    )
)
cells.append(
    nbf.v4.new_markdown_cell(
        """## Part E: final tables and analysis

This cell converts the raw JSON and CSV measurements into the required `METRICS.md` and GPU-hour record. All values come from this run."""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """run_logged([PYTHON, "finalize_hw45.py"], "Part E and final report")
print((ROOT / "METRICS.md").read_text(encoding="utf-8"))"""
    )
)
cells.append(
    nbf.v4.new_markdown_cell("## Measured tables and figures")
)
cells.append(
    nbf.v4.new_code_cell(
        """import pandas as pd
from IPython.display import Image, display

display(pd.read_csv(ROOT / "outputs" / "technique_comparison.csv"))
display(pd.read_csv(ROOT / "outputs" / "serving_benchmark.csv"))
display(pd.read_csv(ROOT / "outputs" / "kv_cache_context.csv"))
display(Image(filename=ROOT / "outputs" / "batch_size_throughput.png"))
display(Image(filename=ROOT / "outputs" / "baseline_confusion_matrix.png"))
display(Image(filename=ROOT / "outputs" / "baseline_training_validation_loss.png"))
display(Image(filename=ROOT / "outputs" / "validation_loss_by_technique.png"))"""
    )
)
cells.append(
    nbf.v4.new_markdown_cell("## Final automatic checklist")
)
cells.append(
    nbf.v4.new_code_cell(
        """training = json.loads((ROOT / "outputs" / "training_summary.json").read_text(encoding="utf-8"))
serving = json.loads((ROOT / "outputs" / "serving_summary.json").read_text(encoding="utf-8"))
metrics_text = (ROOT / "METRICS.md").read_text(encoding="utf-8")
reservation_text = (ROOT / "RESERVATION_RECORD.md").read_text(encoding="utf-8")
checkpoint_text = (ROOT / "MODEL_CHECKPOINT.md").read_text(encoding="utf-8")

required = [
    ROOT / "METRICS.md",
    ROOT / "RUN_LOG.txt",
    ROOT / "AI_USE.md",
    ROOT / "GPU_HOURS.md",
    ROOT / "RESERVATION_RECORD.md",
    ROOT / "outputs" / "gpu_uuid_before.txt",
    ROOT / "outputs" / "gpu_uuid_after.txt",
    ROOT / "outputs" / "batch_size_throughput.png",
    ROOT / "outputs" / "baseline_confusion_matrix.png",
    ROOT / "outputs" / "baseline_training_validation_loss.png",
    ROOT / "outputs" / "validation_loss_by_technique.png",
    ROOT / "outputs" / "technique_comparison.csv",
    ROOT / "outputs" / "serving_benchmark.csv",
    ROOT / "outputs" / "kv_cache_context.csv",
]
checks = {
    "RTX 4090 recorded": "4090" in training["gpu"]["name"],
    "exact SLICE starts the row range": training["dataset_manifest"]["requested_row_range_zero_based"][0] == 834,
    "no more than 5,000 consecutive rows": training["dataset_manifest"]["selected_rows_before_cleaning"] <= 5000,
    "70/15/15 split recorded": training["dataset_manifest"]["split_ratio"] == "70/15/15",
    "first baseline OOM recorded": any(row["status"] == "oom" for row in training["batch_scaling"] if row["technique"] == "baseline"),
    "all five independent/combined sweeps completed": len(training["maximum_batches"]) == 5,
    "gradient accumulation measured": "gradient_accumulation" in training["full_runs"],
    "gradient accumulation used the same optimizer-step count": training["full_runs"]["gradient_accumulation"]["optimizer_steps"] == training["full_runs"]["baseline"]["optimizer_steps"],
    "five worst-class errors diagnosed": len(training["full_runs"]["baseline"].get("five_worst_class_mistakes", [])) == 5,
    "Part C exact skip statement present": "Part C skipped (optional)" in metrics_text,
    "HF plus vLLM 1/8/32 measured": [row["batch_size"] for row in serving["vllm"]["metrics"]] == [1, 8, 32],
    "batch-8 context capacity measured": serving["vllm"]["longest_context_batch_8"] is not None,
    "both UUID captures exist": (ROOT / "outputs" / "gpu_uuid_before.txt").exists() and (ROOT / "outputs" / "gpu_uuid_after.txt").exists(),
    "reservation record filled in": "Fill this" not in reservation_text and all(
        line.split(":", 1)[1].strip()
        for line in reservation_text.splitlines()
        if line.startswith("- ") and ":" in line
    ),
    "large checkpoint link recorded": "pending upload" not in checkpoint_text.lower(),
    "all deliverable files exist": all(path.exists() for path in required),
}
for label, passed in checks.items():
    print(("PASS" if passed else "FAIL") + ": " + label)
assert all(checks.values()), "At least one HW4.5 requirement check failed."
with LOG.open("a", encoding="utf-8") as handle:
    handle.write(f"\\nRun completed: {datetime.now().astimezone().isoformat()}\\n")
    handle.write("HW4.5 automatic checklist passed.\\n")
print("HW4.5 automatic checklist passed.")"""
    )
)

nb.cells = cells
nbf.write(nb, ROOT / "Homework_4_5.ipynb")
print("Created", ROOT / "Homework_4_5.ipynb")
