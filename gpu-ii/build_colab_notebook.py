#!/usr/bin/env python3
"""Build a portable Colab notebook with all assignment scripts embedded."""
from pathlib import Path
import base64
import gzip
import json
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
resources = {}
for path in ROOT.iterdir():
    if path.suffix in (".py", ".md", ".json", ".txt", ".sh") and path.name != Path(__file__).name:
        resources[path.name] = path.read_text()
payload = base64.b64encode(gzip.compress(json.dumps(resources).encode())).decode()
nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"name": "GPU_II_Colab_Run_All.ipynb", "provenance": []},
    "accelerator": "GPU",
}
cells = []
def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))
def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# GPU II: HW4.a and HW4.5

Student: Denisha Ketan Tank

SID4 = 8834, SEED = 8834, SLICE = 834, HP_ID = 2, CLS_A = 4, CLS_B = 6.

Before running, select **Runtime → Change runtime type → GPU**. Then select **Runtime → Run all**.

This notebook contains the supporting Python files. It installs a separate Python 3.11 environment, downloads the Twitter dataset and pretrained models, runs smoke checks, completes HW4.a, and then runs HW4.5. No GitHub credentials, API keys, manual uploads, or runtime restart are required by the notebook. The actual assigned GPU is recorded, including when Colab provides a T4 instead of a 4090.

Part C skipped (optional).

Allow the notebook to finish. Completed stages can be reused if Run all is selected again in the same runtime. Colab can still reclaim or disconnect a session; download the final archive and save this executed notebook when done. The final archive includes the BERT checkpoint, tables, figures, logs, code, and written findings. These are real experiments, so their results are not known in advance.""")

code(r'''from pathlib import Path
from datetime import datetime, timezone
import base64, gzip, hashlib, json, os, platform, shutil, subprocess, sys, time

if platform.system() != "Linux" or not shutil.which("nvidia-smi"):
    raise RuntimeError("Select Runtime > Change runtime type > GPU in Google Colab, then Run all again. A Windows/Mac CPU session cannot run vLLM CUDA benchmarks.")

gpu_listing = subprocess.check_output(["nvidia-smi", "-L"], text=True)
print(gpu_listing)
WORK = Path("/content/gpu-ii") if Path("/content").exists() else Path.cwd() / "gpu-ii-colab-run"
WORK.mkdir(parents=True, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["GPU_ASSIGNMENT_EAGER"] = "1"
LOG = WORK / "COLAB_RUN_LOG.txt"
PYTHON = WORK / ".venv" / "bin" / "python"

def run(command, label, retries=0):
    for attempt in range(retries + 1):
        print(f"\n{label} (attempt {attempt + 1})", flush=True)
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {label}\n")
            handle.write("COMMAND: " + " ".join(map(str, command)) + "\n")
            process = subprocess.Popen(list(map(str, command)), cwd=WORK,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                bufsize=1, env=os.environ.copy())
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
            status = process.wait()
            handle.write(f"EXIT CODE: {status}\n")
        if status == 0:
            return
        if attempt < retries:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{label} failed (exit {status}). Full output is in {LOG}. Completed stages remain saved.")

print("Working directory:", WORK)
print("No installation or model training is performed on your own PC when using a hosted Colab runtime.")
''')

code('''# Restore the embedded assignment files. No GitHub download is needed.
EMBEDDED_PAYLOAD = ''' + repr(payload) + r'''
embedded = json.loads(gzip.decompress(base64.b64decode(EMBEDDED_PAYLOAD)))
for name, contents in embedded.items():
    destination = WORK / name
    if destination.exists() and name in {"METRICS.md", "HW4A_METRICS.md", "hw4a_baseline.json", "RESERVATION_RECORD.md", "MODEL_CHECKPOINT.md", "RUN_LOG.txt", "RUN_LOG_HW4A.txt"}:
        continue
    destination.write_text(contents, encoding="utf-8")

# Fixed versions are isolated from Colab's already-imported packages.
dependencies = [
    "vllm==0.10.2", "torch==2.8.0", "transformers==4.55.4",
    "bitsandbytes==0.47.0", "accelerate>=1.0,<2", "numpy==1.26.4",
    "pandas>=2.2,<3", "scikit-learn>=1.5,<2", "matplotlib>=3.9,<4",
    "seaborn>=0.13,<0.14", "nbformat>=5.10,<6", "kagglehub>=0.3,<1",
    "ipykernel>=6.29,<7", "nbconvert>=7.16,<8",
]
(WORK / "requirements_colab.txt").write_text("\n".join(dependencies) + "\n")
TOOLS = WORK / ".setup-tools"
if not (TOOLS / "uv").exists():
    run([sys.executable, "-m", "pip", "install", "--quiet", "--target", TOOLS, "uv==0.8.22"], "Install environment manager", retries=2)
uv_env = os.environ.copy()
uv_env["PYTHONPATH"] = str(TOOLS)
subprocess.run([sys.executable, "-m", "uv", "venv", "--python", "3.11", WORK / ".venv"],
               env=uv_env, check=True) if not PYTHON.exists() else None
old_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = str(TOOLS)
try:
    run([sys.executable, "-m", "uv", "pip", "install", "--python", PYTHON,
         "-r", WORK / "requirements_colab.txt"], "Install compatible CUDA packages", retries=2)
finally:
    if old_pythonpath is None:
        os.environ.pop("PYTHONPATH", None)
    else:
        os.environ["PYTHONPATH"] = old_pythonpath

probe = subprocess.check_output([str(PYTHON), "-c", "import torch,json; assert torch.cuda.is_available(); print(json.dumps({'gpu':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'torch':torch.__version__,'cuda':torch.version.cuda}))"], cwd=WORK, text=True)
hardware = json.loads(probe.strip().splitlines()[-1])
print(hardware)
if hardware["capability"][0] < 8:
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    print("Using vLLM's Turing-compatible backend for this GPU.")

settings = json.loads((WORK / "settings.json").read_text())
settings["num_workers"] = 0  # Small cached tensor data needs no worker processes in Colab.
(WORK / "settings.json").write_text(json.dumps(settings, indent=2))
identity = hashlib.sha256(json.dumps({"code": EMBEDDED_PAYLOAD, "settings": settings,
    "hardware": hardware, "gpu_uuid": gpu_listing, "dependencies": dependencies}, sort_keys=True).encode()).hexdigest()
identity_file = WORK / "run_identity.txt"
if identity_file.exists() and identity_file.read_text().strip() != identity:
    raise RuntimeError("The saved run used different code, hardware, or settings. Download its results first, then use a fresh Colab runtime for this version.")
identity_file.write_text(identity)
os.environ["GPU_ASSIGNMENT_RESUME"] = "1"
(WORK / "hardware_colab.json").write_text(json.dumps(hardware, indent=2))
versions = subprocess.check_output([str(PYTHON), "-m", "pip", "freeze"], text=True) if subprocess.run([str(PYTHON), "-m", "pip", "--version"], capture_output=True).returncode == 0 else subprocess.check_output([sys.executable, "-m", "uv", "pip", "freeze", "--python", str(PYTHON)], env=uv_env, text=True)
(WORK / "installed_versions.txt").write_text(versions)

def stage(name, command, required):
    marker = WORK / f".complete_{name}.json"
    if marker.exists() and all((WORK / item).exists() for item in required):
        print(f"Already completed: {name}. Reusing the saved measurements.")
        return
    run(command, name)
    absent = [item for item in required if not (WORK / item).exists()]
    if absent:
        raise RuntimeError(f"{name} did not create required files: {absent}")
    marker.write_text(json.dumps({"completed_utc": datetime.now(timezone.utc).isoformat(), "identity": identity}, indent=2))

print("Setup complete. Next: smoke checks and automatic dataset download.")
''')

md("""## Short smoke check and dataset download

The smoke check exercises the CUDA training paths with a tiny temporary model. Synthetic examples are discarded and never used as assignment results. The real experiments below use the Kaggle Twitter sentiment dataset, zero-based rows 834 through 5833, and the same saved 70/15/15 split.""")
code(r'''stage("smoke", [PYTHON, "smoke_test.py", "--gpu"], [])
run([PYTHON, "get_twitter_data.py"], "Download the public Twitter dataset", retries=3)
run([PYTHON, "hw45_training.py", "prepare", "--settings", "settings.json"], "Freeze the real data and split", retries=2)
manifest = json.loads((WORK / "cache" / "dataset_manifest.json").read_text())
print(json.dumps(manifest, indent=2))
''')

md("""## HW4.a: train, evaluate, save, and reload BERT

The baseline uses BERT, batch size 16, learning rate 0.00002, sequence length 128, and three epochs. This stage records the epoch curves, test metrics, class errors, GPU memory, wall time, checkpoint, and predictions from the reloaded model.""")
code(r'''stage("hw4a_training", [PYTHON, "hw4a_training.py", "--settings", "settings.json"],
    ["outputs/hw4a_training_summary.json", "checkpoints/hw4a_bert_classifier/config.json",
     "outputs/hw4a_learning_curves.png", "outputs/hw4a_confusion_matrix.png"])
''')

md("""## HW4.a: Hugging Face and vLLM serving

Qwen2.5-0.5B-Instruct is the second pretrained model. The same model and precision are used for the Hugging Face batch-1 pipeline and vLLM batches 1, 8, and 32. Eager mode reduces Colab startup overhead and is kept the same in HW4.a and HW4.5.""")
code(r'''stage("hw4a_serving", [PYTHON, "hw45_serving.py", "--output-prefix", "hw4a_", "--skip-context-capacity"],
      ["outputs/hw4a_serving_summary.json", "outputs/hw4a_serving_benchmark.csv"])
run([PYTHON, "finalize_hw4a.py"], "Write the HW4.a results and shared baseline")
print((WORK / "HW4A_METRICS.md").read_text())
''')

md("""## HW4.5 Parts A and B: batch scaling and memory techniques

Each probe starts a separate process and records the first CUDA OOM as the expected memory-wall result. Ordinary failures stop with a useful error instead of being labeled as OOM. AMP, checkpointing, and the 8-bit optimizer are tested independently, then AMP plus the 8-bit optimizer is tested as a candidate combination. Gradient accumulation uses the same effective batch and optimizer-step budget as the full-precision comparison.

This is the longest stage. Individual completed probes and full training runs are saved for resuming in this runtime.""")
code(r'''stage("hw45_training", [PYTHON, "hw45_training.py", "all", "--settings", "settings.json"],
    ["outputs/training_summary.json", "outputs/batch_scaling.csv", "outputs/technique_comparison.csv",
     "outputs/batch_size_throughput.png"])
''')
md("""## Part C

Part C skipped (optional).

## Part D: repeat serving and examine context capacity

This stage repeats the serving measurements and tests batch-8 context lengths. Active KV-cache size is calculated from the actual model configuration. Total process memory is also measured; vLLM reserves a pool in advance, so allocated process memory and active cache size are different quantities. If every context fits, the reported value is the longest tested/model-supported context, not a proven memory exhaustion boundary.""")
code(r'''stage("hw45_serving", [PYTHON, "hw45_serving.py"],
    ["outputs/serving_summary.json", "outputs/serving_benchmark.csv", "outputs/kv_cache_context.csv",
     "outputs/gpu_uuid_after.txt"])
run([PYTHON, "finalize_hw45.py"], "Write HW4.5 findings and GPU-hour calculation")

# Record the actual hardware label in generated prose and plot titles.
# A Colab T4 must not be described as an RTX 4090.
for name in ("METRICS.md", "README.md", "RUN_LOG.txt"):
    path = WORK / name
    if path.exists():
        text = path.read_text().replace("RTX 4090", hardware["gpu"])
        path.write_text(text)
print((WORK / "METRICS.md").read_text())
''')

md("""## Tables and figures

The generated reports and tables below are based on the actual run. Read the five error diagnoses and the recommendations before submitting: these are draft interpretations that should agree with the examples and measurements.""")
code(r'''import pandas as pd
from IPython.display import Image, Markdown, display

for name in ("hw4a_epoch_metrics.csv", "hw4a_per_class_metrics.csv", "hw4a_custom_inference.csv",
             "technique_comparison.csv", "serving_benchmark.csv", "kv_cache_context.csv"):
    print(name)
    display(pd.read_csv(WORK / "outputs" / name))
for name in ("hw4a_learning_curves.png", "hw4a_confusion_matrix.png", "batch_size_throughput.png",
             "validation_loss_by_technique.png"):
    display(Image(filename=str(WORK / "outputs" / name)))
''')

md("""## Verify and package the results

The archive includes the full BERT checkpoint so it can be uploaded with the results or shared separately. Colab does not require a lab reservation; actual run timestamps and GPU evidence are recorded automatically. Download this executed notebook with **File → Download → Download .ipynb** as well. A passing computational check does not replace reviewing the written analysis or publishing the final repository tag.""")
code(r'''training = json.loads((WORK / "outputs" / "training_summary.json").read_text())
baseline = json.loads((WORK / "outputs" / "hw4a_training_summary.json").read_text())
serving_result = json.loads((WORK / "outputs" / "serving_summary.json").read_text())
prior = json.loads((WORK / "hw4a_baseline.json").read_text())
checks = {
    "HW4.a and HW4.5 used the same dataset hash": baseline["dataset_manifest"]["source_sha256"] == training["dataset_manifest"]["source_sha256"],
    "HW4.a and HW4.5 used identical data settings": baseline["dataset_manifest"]["preparation_settings"] == training["dataset_manifest"]["preparation_settings"],
    "source rows begin at 834": manifest["requested_row_range_zero_based"][0] == 834,
    "fixed 70/15/15 split": manifest["split_ratio"] == "70/15/15",
    "BERT saved and reloaded": len(baseline["reloaded_custom_inference"]) == 2,
    "original measured baseline populated": prior["training_wall_seconds"] > 0,
    "five worst-class HW4.a mistakes available": len(baseline["training"]["five_worst_class_mistakes"]) == 5,
    "all technique sweeps reached OOM": all(any(row["technique"] == name and row["status"] == "oom" for row in training["batch_scaling"]) for name in training["maximum_batches"]),
    "same accumulation optimizer-step count": training["full_runs"]["gradient_accumulation"]["optimizer_steps"] == training["full_runs"]["baseline"]["optimizer_steps"],
    "vLLM batches 1,8,32 measured": [row["batch_size"] for row in serving_result["vllm"]["metrics"]] == [1,8,32],
    "context results exist": len(serving_result["vllm"]["context_results"]) > 0,
    "optional Part C skip recorded": "Part C skipped (optional)" in (WORK / "METRICS.md").read_text(),
    "both GPU UUID captures exist": all((WORK / "outputs" / name).exists() for name in ("gpu_uuid_before.txt", "gpu_uuid_after.txt")),
}
for label, passed in checks.items():
    print(("PASS" if passed else "FAIL") + ": " + label)
(WORK / "verification.json").write_text(json.dumps(checks, indent=2))
assert all(checks.values()), "A requirement check failed. The measurements remain available for inspection."

end = datetime.now(timezone.utc).isoformat()
start_marker = json.loads((WORK / ".complete_smoke.json").read_text())["completed_utc"]
(WORK / "RESERVATION_RECORD.md").write_text(
    "# Runtime record\n\nProvider: Google Colab hosted GPU runtime, or Linux GPU runtime when executed locally.\n"
    "Lab reservation: not applicable for this hosted/personal runtime.\n"
    f"GPU: {hardware['gpu']}\n\nFirst completed smoke check (UTC): {start_marker}\n\nCompleted (UTC): {end}\n\n"
    + gpu_listing)
(WORK / "MODEL_CHECKPOINT.md").write_text(
    "# Saved BERT checkpoint\n\nThe full checkpoint and tokenizer are included in this results archive under "
    "`gpu-ii/checkpoints/hw4a_bert_classifier/`. The saved model was reloaded and tested.\n\n"
    "Training command: `.venv/bin/python hw4a_training.py --settings settings.json`.\n\n"
    "Before repository submission, upload this large checkpoint to approved storage or Git LFS and record its shareable link here.\n")
for source, target in (("COLAB_RUN_LOG.txt", "RUN_LOG.txt"), ("COLAB_RUN_LOG.txt", "RUN_LOG_HW4A.txt")):
    shutil.copy2(WORK / source, WORK / target)

import zipfile
archive = WORK.parent / "GPU_II_RESULTS.zip"
excluded_parts = {".venv", ".setup-tools", "cache", "__pycache__", ".ipynb_checkpoints"}
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zipped:
    for path in WORK.rglob("*"):
        relative = path.relative_to(WORK)
        if path.is_file() and not any(part in excluded_parts for part in relative.parts) and path.name != "Twitter_Data.csv":
            zipped.write(path, Path("gpu-ii") / relative)
print("COMPUTATIONAL CHECKLIST PASSED")
print(f"Archive: {archive} ({archive.stat().st_size / 2**20:.1f} MiB)")
print("Save the executed notebook too: File > Download > Download .ipynb.")
try:
    from google.colab import files
    files.download(str(archive))
except ImportError:
    from IPython.display import FileLink
    display(FileLink(str(archive)))
''')

nb.cells = cells
destination = ROOT / "GPU_II_Colab_Run_All.ipynb"
if destination.exists():
    existing = nbf.read(destination, as_version=4)
    if any(cell.get('execution_count') is not None or cell.get('outputs') for cell in existing.cells if cell.cell_type == 'code'):
        raise SystemExit("Refusing to overwrite the executed submission notebook. Run the existing notebook in Colab instead.")
nbf.write(nb, destination)
print("Created", destination)

if __name__ == "__main__":
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(destination), "exec")
    nbf.validate(nb)
    print("Notebook schema and every code cell are valid.")
