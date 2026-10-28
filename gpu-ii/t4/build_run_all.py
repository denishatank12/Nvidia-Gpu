"""Build a fresh T4 notebook; never overwrite historical executed results."""
from pathlib import Path
import base64
import gzip
import hashlib
import json
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
NAMES = ['hw45_training.py', 'hw4a_training.py', 'hw45_serving.py', 'get_twitter_data.py',
         'smoke_test.py', 'accurate_serving.py', 'test_measurements.py', 'test_reporting.py', 'test_precision.py', 'run_experiments.py', 'make_reports.py']
resources = {name: (ROOT / name).read_text() for name in NAMES}
resources['settings.json'] = json.dumps({'data_path': 'Twitter_Data.csv', 'text_column': 'clean_text',
    'label_column': 'category', 'model_name': 'bert-base-uncased', 'max_length': 128,
    'start_row': 834, 'row_limit': 5000, 'seed': 8834, 'epochs': 3,
    'learning_rate': 0.00002, 'weight_decay': 0.01, 'original_batch': 16,
    'max_probe_batch': 65536, 'num_workers': 0}, indent=2)
payload = base64.b64encode(gzip.compress(json.dumps(resources).encode(), mtime=0)).decode()
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md('''# GPU II: complete HW4.a and HW4.5 on a T4

Denisha Ketan Tank

SID4 = 8834, SEED = 8834, SLICE = 834, HP_ID = 2, CLS_A = 4, CLS_B = 6.

This is a fresh experiment notebook, not the old T4 notebook with edited scores. It is designed for one NVIDIA T4 with approximately 15 GiB usable VRAM. It records the GPU actually used and runs the full dataset slice and full training epochs. T4 mixed precision uses FP16 and loss scaling, not unsupported BF16. The FP32 baseline remains unchanged.

**In Google Colab, choose Runtime > Change runtime type > T4 GPU > Save, then Runtime > Run all.** This is a single self-contained file. Do not upload Python scripts or the dataset separately. It installs a private Python 3.11 environment without replacing Colab's kernel and downloads the exact Twitter data and both models automatically. Internet access and at least 20 GiB free disk space are needed. Linux or WSL2 Jupyter on a T4 is also supported; native Windows is not supported by vLLM.

Installation, smoke tests, data download, both assignments, reports, verification, and archiving run in order. Wait for both real-GPU smoke stages to pass before leaving the notebook running. A successful run produces a results ZIP; save this executed notebook too. No duration or accuracy is invented in advance. Colab can disconnect or reclaim the GPU: this notebook cannot guarantee an uninterrupted overnight allocation. Resuming reuses completed stages only while the same run folder and GPU remain available. Download the ZIP promptly at completion; Colab runtime files are temporary.

Part C skipped (optional).

Expected OOM probes are caught and recorded. BERT's warning about newly initialized classifier weights is also expected: those are the weights fine-tuning learns. Other failures stop with a log instead of being disguised as successful measurements. Completed stages can resume only with the same code, GPU, and settings.''')

code(r'''from pathlib import Path
from datetime import datetime, timezone
import base64, gzip, hashlib, json, os, platform, shutil, subprocess, sys, time

if platform.system() != 'Linux':
    raise RuntimeError('Use Google Colab with a T4 GPU, or Linux/WSL2 Jupyter on a T4. Native Windows and macOS cannot run this vLLM CUDA notebook.')
if platform.machine().lower() not in ('x86_64', 'amd64') or not shutil.which('nvidia-smi'):
    raise RuntimeError('This notebook needs an x86-64 Linux/WSL NVIDIA GPU runtime with nvidia-smi available.')
if not any(shutil.which(name) for name in ('cc', 'gcc', 'clang')):
    raise RuntimeError('A C compiler is needed for CUDA/Triton runtime launchers. Install build-essential in Linux/WSL once, then Run All again.')
GPU_LIST = subprocess.check_output(['nvidia-smi', '-L'], text=True)
GPU_LINE = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits'], text=True).splitlines()[0]
print(GPU_LIST)
print('GPU 0 name, UUID, total/free MiB, driver:', GPU_LINE)
if 'T4' not in GPU_LINE.upper():
    raise RuntimeError('This is the T4-specific notebook. In Colab choose Runtime > Change runtime type > T4 GPU > Save, then Run all. No measurements have started.')
if shutil.disk_usage(Path.cwd()).free < 20 * 2**30:
    raise RuntimeError('Free at least 20 GiB disk space before installing the CUDA environment and downloading models.')
os.environ.update({'CUDA_VISIBLE_DEVICES': '0', 'TOKENIZERS_PARALLELISM': 'false',
    'PYTHONUNBUFFERED': '1', 'MPLBACKEND': 'Agg', 'HF_HUB_DOWNLOAD_TIMEOUT': '180',
    'HF_HUB_ETAG_TIMEOUT': '60', 'VLLM_USE_V1': '0', 'GPU_ASSIGNMENT_EAGER': '1',
    'GPU_ASSIGNMENT_RESUME': '1', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1',
    'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'UV_NO_CACHE': '1',
    'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
''')

code('EMBEDDED_PAYLOAD = ' + repr(payload) + r'''
embedded = json.loads(gzip.decompress(base64.b64decode(EMBEDDED_PAYLOAD)))
gpu_identity = ','.join(part.strip() for i, part in enumerate(GPU_LINE.split(',')) if i != 3)
identity = hashlib.sha256((EMBEDDED_PAYLOAD + gpu_identity).encode()).hexdigest()
BASE = Path('/content') if Path('/content').exists() else Path.cwd()
WORK = BASE / ('GPU_II_T4_RUN_' + identity[:12])
WORK.mkdir(parents=True, exist_ok=True)
LOG = WORK / 'RUN_LOG.txt'
PYTHON = WORK / '.venv' / 'bin' / 'python'
for name, contents in embedded.items():
    path = WORK / name
    if name == 'settings.json' and path.exists():
        continue  # Preserve the common baseline batch selected by the real VRAM probe.
    path.write_text(contents, encoding='utf-8')
metadata_path = WORK / 'runtime_metadata.json'
if not metadata_path.exists():
    metadata_path.write_text(json.dumps({'started_utc': datetime.now(timezone.utc).isoformat(),
        'gpu_listing': GPU_LIST, 'gpu_start': GPU_LINE, 'platform': platform.platform(),
        'identity': identity, 'provider': 'Google Colab' if Path('/content').exists() else 'local Linux/WSL GPU runtime'}, indent=2))

def run(command, label, retries=0):
    for attempt in range(retries + 1):
        print('\n' + label + f' (attempt {attempt + 1})', flush=True)
        start = time.monotonic()
        with LOG.open('a', encoding='utf-8') as handle:
            handle.write(f'\n[{datetime.now(timezone.utc).isoformat()}] {label}\n')
            handle.write('COMMAND: ' + ' '.join(map(str, command)) + '\n')
            handle.flush()
            process = subprocess.Popen(list(map(str, command)), cwd=WORK, env=os.environ.copy(),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True); handle.write(line); handle.flush()
            result = process.wait()
            handle.write(f'EXIT CODE: {result}\n'); handle.flush()
        if result == 0: return time.monotonic() - start
        if attempt < retries: time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'{label} failed, exit {result}. Read {LOG}. Earlier completed stages are saved; no failed stage is marked complete.')

def stage(name, command, required):
    marker = WORK / ('.complete_' + name + '.json')
    if marker.exists() and all((WORK / n).exists() for n in required):
        data = json.loads(marker.read_text())
        if data['identity'] != identity: raise RuntimeError('Refusing to reuse measurements from a different run.')
        print('Already completed, reusing:', name); return
    seconds = run(command, name, retries=1)
    missing = [n for n in required if not (WORK / n).exists()]
    if missing: raise RuntimeError(f'{name} did not create {missing}')
    marker.write_text(json.dumps({'identity': identity, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'command_wall_seconds': seconds}, indent=2))

print('All run files and resumable stages will be stored in:', WORK)
''')

md('''## Install an isolated, pinned GPU environment

The notebook kernel is not replaced and does not need a restart. Python 3.11 and the GPU packages are installed in the run folder. The NVIDIA driver is a machine prerequisite, not something this notebook changes. No administrator access is used.''')
code(r'''TOOLS = WORK / '.setup_tools'
if not (TOOLS / 'uv').exists():
    run([sys.executable, '-m', 'pip', 'install', '--quiet', '--target', TOOLS, 'uv==0.8.22'], 'Install uv environment manager', retries=2)
dependencies = ['vllm==0.10.2', 'torch==2.8.0', 'transformers==4.55.4',
    'bitsandbytes==0.47.0', 'accelerate==1.10.1', 'numpy==1.26.4', 'pandas==2.2.3',
    'scikit-learn==1.7.2', 'matplotlib==3.10.7', 'seaborn==0.13.2',
    'nbformat==5.10.4', 'kagglehub==0.3.13', 'ipykernel==6.30.1', 'nbconvert==7.16.6']
(WORK / 'requirements_locked.txt').write_text('\n'.join(dependencies) + '\n')
previous_path = os.environ.get('PYTHONPATH')
os.environ['PYTHONPATH'] = str(TOOLS)
try:
    if not PYTHON.exists():
        run([sys.executable, '-m', 'uv', 'venv', '--python', '3.11', WORK / '.venv'], 'Create isolated Python 3.11 environment', retries=2)
    run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', PYTHON, '-r', WORK / 'requirements_locked.txt'], 'Install pinned CUDA packages', retries=2)
    versions = subprocess.check_output([sys.executable, '-m', 'uv', 'pip', 'freeze', '--python', PYTHON], text=True)
finally:
    if previous_path is None: os.environ.pop('PYTHONPATH', None)
    else: os.environ['PYTHONPATH'] = previous_path
(WORK / 'installed_versions.txt').write_text(versions)
run([PYTHON, '-c', "import torch; assert torch.cuda.is_available(), 'CUDA is not visible. Check the host NVIDIA driver and WSL GPU support.'; print('Torch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU:',torch.cuda.get_device_name(0),'VRAM GiB:',torch.cuda.get_device_properties(0).total_memory/2**30)"], 'Verify the actual CUDA GPU')
''')

md('''## Smoke tests before expensive work

These tests use temporary synthetic data and tiny BERT weights, which are never reported as assignment results. They check forward/backward training, AMP, checkpointing, accumulation, the 8-bit optimizer, cache reuse, real vLLM timestamps for batches 1/8/32, and the real cache/scheduler interface. A dependency or API failure is caught here rather than hidden in an overnight run.''')
code(r'''stage('precision_regression', [PYTHON, 'test_precision.py', '--gpu'], [])
stage('reporting_smoke', [PYTHON, 'test_reporting.py'], [])
stage('bert_smoke', [PYTHON, 'smoke_test.py', '--gpu'], [])
stage('measurement_smoke', [PYTHON, 'test_measurements.py', '--gpu'], [])
print('Both real-GPU smoke stages passed. Starting the real dataset experiments.')
''')

md('''## Data preparation and common training batch

The exact same Twitter dataset hash as HW4.a is enforced. We use source rows 834–5833, seed 8834, and the same stratified 70/15/15 split for every experiment. Tokenization happens once and split indices are saved. Batch 16 is attempted first; a lower common batch is selected only if a real FP32 OOM requires it. Both assignments then use the selected baseline settings.''')
code(r'''stage('data', [PYTHON, 'run_experiments.py', 'prepare'], ['cache/dataset_manifest.json', 'cache/twitter_bert_cache.pt', 'settings.json'])
settings = json.loads((WORK / 'settings.json').read_text())
manifest = json.loads((WORK / 'cache/dataset_manifest.json').read_text())
print('Selected settings:', json.dumps(settings, indent=2))
print('Dataset and split:', json.dumps(manifest, indent=2))
''')

md('''## HW4.a: full BERT fine-tuning and first serving benchmark

BERT is trained for three full epochs with cross-entropy, AdamW, learning rate 0.00002, and sequence length 128. The notebook records losses, validation accuracy, test metrics, class errors, peak GPU allocation and wall time, then saves and reloads the classifier for two custom examples.

The second model is Qwen2.5-0.5B-Instruct. HF batch 1 and vLLM batches 1/8/32 all use FP16, identical prompt families and exactly 64 generated tokens. Each vLLM request has its own recorded completion time; the batch duration is used only for aggregate throughput.''')
code(r'''stage('hw4a_training', [PYTHON, 'hw4a_training.py', '--settings', 'settings.json'],
    ['outputs/hw4a_training_summary.json', 'checkpoints/hw4a_bert_classifier/model.safetensors', 'outputs/hw4a_learning_curves.png'])
stage('hw4a_serving', [PYTHON, 'accurate_serving.py', 'benchmark', '--prefix', 'hw4a_'],
    ['outputs/hw4a_serving_summary.json', 'outputs/hw4a_request_timings.csv'])
print('HW4.a training and serving stages finished.')
''')

md('''## HW4.5 Parts A and B: memory walls and training comparisons

First, each memory technique is tested independently at power-of-two batch sizes until a real OOM. Up to the training-set size, successful points are full training epochs. Larger capacity-only probes repeat existing training rows to materialize the entire requested physical batch for one real forward/backward/optimizer update. These are labelled separately, never used for accuracy, and never called full epochs. Final quality runs use the largest successful power-of-two batch that fits the original training split and train for three full epochs.

Next, a separate controlled comparison trains all techniques with the same effective batch, examples, three epochs, seed, and optimizer-update budget. Accumulation uses a smaller physical batch. AMP uses T4-compatible FP16 with conservative loss scaling; it does not request BF16. A recoverable FP16 overflow replays the same logical update with a lower loss scale and restored dropout RNG; weights and update counts change only after a successful update. Retries and their actual execution costs are recorded. Persistent numerical failure stops rather than being hidden. This is the fair quality comparison; changing the maximum batch does not masquerade as a technique-only accuracy effect. The existing baseline is reused when it is exactly the same experiment.''')
code(r'''stage('hw45_training', [PYTHON, 'hw45_training.py', 'all', '--settings', 'settings.json'],
    ['outputs/training_summary.json', 'outputs/batch_scaling.csv', 'outputs/technique_comparison.csv'])
stage('controlled_training', [PYTHON, 'run_experiments.py', 'controlled'], ['outputs/controlled_training.json'])
print('All full-epoch memory probes, maximum-batch fine-tunes, and controlled fine-tunes finished.')
''')

md('''## Part C

Part C skipped (optional).

## Part D: repeated serving and observed KV-cache capacity

The serving benchmark is repeated with the same model, precision, prompt families, output length, and per-request timing definition. There are at least 100 requests per configuration and at least 20 measured vLLM batches per batch size, after warm-up.

Context probes use fresh GPU processes. Actual cache tensor storage and scheduler block occupancy are measured. Success requires eight full prompts to reside together; serial completion of eight queued requests is not called simultaneous batch-8 capacity. The maximum supported model length is never silently extended. Memory exhaustion, a cache allocation limit, admission pressure, and a model-position ceiling are distinguished. A failing capacity bracket is refined to within 128 prompt tokens, not mislabelled as an exact one-token threshold.''')
code(r'''stage('hw45_serving', [PYTHON, 'accurate_serving.py', 'benchmark'], ['outputs/serving_summary.json', 'outputs/request_timings.csv'])
stage('context_capacity', [PYTHON, 'accurate_serving.py', 'context'], ['outputs/context_summary.json', 'outputs/kv_cache_context.csv'])
subprocess.run(['nvidia-smi', '-L'], stdout=(WORK / 'outputs/gpu_uuid_after.txt').open('w'), check=True, text=True)
print('Repeated serving and observed context-capacity measurements finished.')
''')

md('''## Findings, comparisons, and figures

The reports below are generated from this run's measurements. All four independent techniques are ranked using the matched-budget controls. Configuration recommendations use validation accuracy, not repeated tuning on test accuracy. Error diagnoses are cautious interpretations of the actual misclassified tweets; read them before submitting work in your own name.''')
code(r'''run([PYTHON, 'make_reports.py'], 'Generate findings and verify real measured artifacts')
from IPython.display import display, Markdown, Image
for name in ['HW4A_METRICS.md', 'METRICS.md', 'AI_USE.md', 'GPU_HOURS.md']:
    display(Markdown((WORK / name).read_text()))
for name in ['hw4a_learning_curves.png', 'hw4a_confusion_matrix.png', 'batch_size_throughput.png', 'controlled_losses.png']:
    display(Image(filename=str(WORK / 'outputs' / name)))
''')

md('''## Verify and package everything

The archive contains the full BERT checkpoint and tokenizer, source code, locked environment versions, dataset manifest and split cache, raw request timings, context-probe logs, plots, reports, GPU UUID captures, and the timestamped console log. It excludes the installed environment and downloaded package caches.

This does not upload anything, create a GitHub tag, invent a lab reservation, or change repository visibility. Save this executed notebook after completion. The large ZIP and the executed notebook are the files to return for the final audit and repository publication.''')
code(r'''verification = json.loads((WORK / 'outputs/verification.json').read_text())
assert verification['all_passed'], 'Keep the log: a measured-artifact verification did not pass.'
metadata = json.loads(metadata_path.read_text())
completed = datetime.now(timezone.utc)
elapsed = (completed - datetime.fromisoformat(metadata['started_utc'])).total_seconds()
(WORK / 'RESERVATION_RECORD.md').write_text(
    '# Actual runtime record\n\nProvider: ' + metadata['provider'] + '\n\n' + GPU_LIST +
    '\nRun started (UTC): ' + metadata['started_utc'] + '\n\nRun completed (UTC): ' + completed.isoformat() +
    '\n\nElapsed session wall time, including setup, downloads, and possible idle/resume intervals: ' + str(round(elapsed, 2)) + ' seconds.\n\n' +
    'No lab booking is invented. A personal/hosted GPU does not need a lab reservation; if this was a reserved lab system, add its actual booking confirmation before submission.\n')
shutil.copy2(LOG, WORK / 'RUN_LOG_HW4A.txt')
(WORK / 'RUN_IDENTITY.json').write_text(json.dumps({'identity': identity, 'completed_utc': completed.isoformat(), 'gpu': GPU_LIST}, indent=2))
import zipfile
archive = BASE / ('GPU_II_T4_RESULTS_' + identity[:12] + '.zip')
excluded = {'.venv', '.setup_tools', '__pycache__', '.ipynb_checkpoints', '.git'}
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as bundle:
    for path in WORK.rglob('*'):
        relative = path.relative_to(WORK)
        if path.is_file() and not any(part in excluded for part in relative.parts):
            bundle.write(path, Path('gpu-ii') / relative)
with zipfile.ZipFile(archive) as bundle:
    bad = bundle.testzip()
    assert bad is None, 'Archive integrity failure: ' + str(bad)
print('MEASURED EXPERIMENTS AND ARTIFACT CHECKS PASSED')
print('Actual GPU:', GPU_LIST.strip())
print('Results archive:', archive)
print('Archive size MiB:', round(archive.stat().st_size / 2**20, 1))
print('SAVE THIS EXECUTED NOTEBOOK as well. Then return the notebook and archive for the final audit/upload.')
try:
    from google.colab import files
except ImportError:
    from IPython.display import FileLink
    display(FileLink(str(archive.relative_to(Path.cwd()))))
else:
    try: files.download(str(archive))
    except Exception as error: print('Automatic browser download was unavailable:', str(error), '\nThe archive remains at', archive)
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python'}, 'accelerator': 'GPU',
               'colab': {'name': 'GPU_II_T4_Run_All.ipynb', 'provenance': []}}
for cell in nb.cells:
    if cell.cell_type == 'code': compile(cell.source, '<notebook cell>', 'exec')
nbf.validate(nb)
destination = ROOT / 'GPU_II_T4_Run_All.ipynb'
if destination.exists():
    prior = nbf.read(destination, as_version=4)
    if any(c.get('outputs') or c.get('execution_count') is not None for c in prior.cells if c.cell_type == 'code'):
        raise SystemExit('Refusing to overwrite an executed T4 notebook.')
nbf.write(nb, destination)
print('Created and validated:', destination)
