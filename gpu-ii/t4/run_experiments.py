"""GPU II orchestration; each additional training condition gets a fresh process."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
EXPECTED_DATA_HASH = "5a7af1ec5fc749ec8e5ea13c574aeb5c06254aa1c081e3421868079d5356b3f4"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(path).with_name(Path(path).name + '.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(command):
    subprocess.run([str(c) for c in command], cwd=ROOT, check=True)


def prepare():
    import hw45_training as training
    run([sys.executable, ROOT / "get_twitter_data.py"])
    if training.file_sha256(ROOT / "Twitter_Data.csv") != EXPECTED_DATA_HASH:
        raise RuntimeError("The CSV differs from the original HW4.a dataset. Do not silently substitute a new version.")
    settings = training.parse_settings(str(ROOT / "settings.json"))
    training.prepare_data(settings)
    # The selection probe is a full, seeded one-epoch training run, reused in
    # the later batch sweep rather than discarded or treated as final accuracy.
    for size in (16, 8, 4, 2):
        path = OUT / f"probe_baseline_{size}.json"
        result = training.probe_subprocess(ROOT / "settings.json", "baseline", size, path)
        if result["status"] == "success":
            settings.original_batch = size
            write(ROOT / "settings.json", dataclasses.asdict(settings))
            print("Selected common FP32 training batch:", size, flush=True)
            return
        if result["status"] != "oom":
            raise RuntimeError("The baseline probe failed for a reason other than memory: " + str(result))
    raise RuntimeError("BERT cannot fit even batch 2. Close other GPU programs or use a GPU with more free VRAM.")


def worker(name, physical, accumulation, destination):
    import hw45_training as training
    settings = training.parse_settings(str(ROOT / "settings.json"))
    result = training.run_training(settings, "baseline" if name == "gradient_accumulation" else name,
                                   physical, settings.epochs, accumulation_steps=accumulation)
    result["technique"] = name
    write(destination, result)


def controlled():
    import hw45_training as training
    original = read(OUT / "hw4a_training_summary.json")
    settings = training.parse_settings(str(ROOT / "settings.json"))
    base = settings.original_batch
    controls = {"baseline": original["training"]}
    for name in ("mixed_precision", "gradient_checkpointing", "optimizer_8bit", "best_combination", "gradient_accumulation"):
        path = OUT / ("controlled_" + name + ".json")
        physical, accumulation = (max(1, base // 4), min(4, base)) if name == "gradient_accumulation" else (base, 1)
        assert physical * accumulation == base
        maximum_path = OUT / ('full_result_' + name + '.json')
        if not path.exists() and maximum_path.exists():
            existing = read(maximum_path)
            if existing['physical_batch'] == physical and existing['accumulation_steps'] == accumulation and existing['epochs'] == settings.epochs:
                write(path, existing)
                print('Reusing an exactly identical completed maximum-batch run for control:', name)
        if not path.exists():
            run([sys.executable, __file__, "worker", "--name", name, "--physical", physical,
                 "--accumulation", accumulation, "--destination", path])
        controls[name] = read(path)
        assert all(controls[name][key] == controls["baseline"][key]
                   for key in ("effective_batch", "optimizer_steps", "samples_seen", "epochs")), name
    write(OUT / "controlled_training.json", {"gpu": original["gpu"], "settings": dataclasses.asdict(settings), "runs": controls})
    print("Matched-budget training complete. Each condition has the same effective batch and update count.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["prepare", "controlled", "worker"])
    p.add_argument("--name")
    p.add_argument("--physical", type=int)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--destination")
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.task == "prepare": prepare()
    elif args.task == "controlled": controlled()
    else: worker(args.name, args.physical, args.accumulation, args.destination)
