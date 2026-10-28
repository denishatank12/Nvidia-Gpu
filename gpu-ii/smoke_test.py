#!/usr/bin/env python3
"""Short checks in temporary storage. No assignment results are overwritten.

python smoke_test.py              # offline CPU checks
python smoke_test.py --gpu        # also exercise BERT + AMP on CUDA
python smoke_test.py --gpu --vllm # also download/load the serving model
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

import hw45_training as training
import hw45_serving as serving


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--vllm", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for path in root.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    for path in root.glob("*.ipynb"):
        import nbformat
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
    print("PASS: Python scripts and notebook cells are valid")

    with tempfile.TemporaryDirectory(prefix="gpu-ii-smoke-") as directory:
        temp = Path(directory)
        model_dir = temp / "tiny_bert"
        model_dir.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "this", "movie", "is", "good", "bad", "okay"]
        (model_dir / "vocab.txt").write_text("\n".join(vocab) + "\n")
        tokenizer = BertTokenizerFast(vocab_file=str(model_dir / "vocab.txt"))
        tokenizer.save_pretrained(model_dir)
        torch.manual_seed(8834)
        model = BertForSequenceClassification(BertConfig(
            vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32, num_labels=3,
        ))
        batch = tokenizer(["this movie is good", "this movie is bad"], padding=True, return_tensors="pt")
        loss = model(**batch, labels=torch.tensor([2, 0])).loss
        loss.backward()
        assert torch.isfinite(loss)
        model.save_pretrained(model_dir)
        reloaded = BertForSequenceClassification.from_pretrained(model_dir).eval()
        with torch.no_grad():
            assert reloaded(**batch).logits.shape == (2, 3)
        print("PASS: real tiny BERT forward/backward, checkpoint save/reload, tokenizer")

        data_path = temp / "synthetic.csv"
        pd.DataFrame({"clean_text": ["this movie is good"] * 6000,
                      "category": [i % 3 for i in range(6000)]}).to_csv(data_path, index=False)
        settings = training.Settings(data_path=str(data_path), model_name=str(model_dir),
                                     max_length=16, num_workers=0)
        with patch.multiple(training, CACHE_DIR=temp / "cache", OUTPUT_DIR=temp / "outputs",
                            CHECKPOINT_DIR=temp / "checkpoints"):
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = training.prepare_data(settings)
            cache = training.load_cache()
            assert manifest["requested_row_range_zero_based"] == [834, 5833]
            assert manifest["split_sizes"] == {"train": 3500, "validation": 750, "test": 750}
            splits = [set(value.tolist()) for value in cache["splits"].values()]
            assert not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
            original_indices = {key: value.clone() for key, value in cache["splits"].items()}
            with contextlib.redirect_stdout(io.StringIO()):
                training.prepare_data(settings)
            assert all(torch.equal(value, training.load_cache()["splits"][key]) for key, value in original_indices.items())
            with contextlib.redirect_stdout(io.StringIO()):
                training.prepare_data(replace(settings, max_length=12))
            assert training.load_cache()["input_ids"].shape == (5000, 12)
            print("PASS: exact row range, 3500/750/750 disjoint split, cache reuse and invalidation")
            loader = training.make_loader(cache, "train", 16, settings, shuffle=True)
            assert next(iter(loader))[0].shape == (16, 16)
            print("PASS: TensorDataset and DataLoader")
            capacity_loader = training.make_loader(cache, 'train', 8192, settings,
                                                   shuffle=True, capacity_batch=True)
            batches = list(capacity_loader)
            assert len(batches) == 1 and batches[0][0].shape == (8192, 16)
            assert len(loader.dataset) == 3500  # The normal split was not enlarged.
            assert batches[0][2].shape == (8192,)
            print('PASS regression: batch 8192 really contains 8192 examples, not a clamped 3500; normal training split unchanged')
            (temp / "outputs").mkdir(exist_ok=True)
            with patch.object(training, "probe_subprocess", return_value={"status": "error", "error": "test failure"}):
                try:
                    training.run_sweep(settings, temp / "settings.json", "baseline")
                except RuntimeError as error:
                    assert "test failure" in str(error)
                else:
                    raise AssertionError("Ordinary errors must not be treated as OOM")
            print("PASS: ordinary probe errors stop the sweep")
            config = type("Config", (), {"num_hidden_layers": 4, "num_attention_heads": 4,
                                         "num_key_value_heads": 2, "hidden_size": 32})()
            assert serving.kv_bytes_per_token(config) == 256
            assert serving.percentile([1, 2, 3], 50) == 2
            print("PASS: KV-cache arithmetic and latency percentile")
            if not args.gpu:
                # Execute the production oversized-batch branch on CPU. Only
                # device plumbing and CUDA counters are mocked; real BERT,
                # checkpointed autograd, AdamW and batch construction run.
                short = replace(settings, start_row=0, row_limit=60, epochs=1, original_batch=2)
                with contextlib.redirect_stdout(io.StringIO()):
                    training.prepare_data(short, force=True)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(training, 'require_cuda'))
                    stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda self, *a, **k: self))
                    stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda self, *a, **k: self))
                    for name in ('synchronize', 'empty_cache', 'reset_peak_memory_stats'):
                        stack.enter_context(patch.object(torch.cuda, name))
                    for name in ('max_memory_allocated', 'max_memory_reserved'):
                        stack.enter_context(patch.object(torch.cuda, name, return_value=0))
                    stack.enter_context(patch.object(training, 'build_optimizer',
                        lambda m, s, q: torch.optim.AdamW(m.parameters(), lr=s.learning_rate)))
                    result = training.run_training(short, 'gradient_checkpointing', 128, 1, memory_probe=True)
                    accumulated = training.run_training(short, 'baseline', 4, 2, accumulation_steps=2)
                    direct = training.run_training(short, 'baseline', 8, 2)
                assert result['observed_first_batch'] == result['samples_seen'] == 128
                assert result['optimizer_steps'] == 1 and 'test_accuracy' not in result
                assert accumulated['samples_seen'] == direct['samples_seen'] == 84
                assert accumulated['optimizer_steps'] == direct['optimizer_steps'] == 12
                assert accumulated['effective_batch'] == direct['effective_batch'] == 8
                print('PASS: production oversized-batch branch with real checkpointed BERT backward and AdamW on CPU (CUDA counters mocked, not GPU evidence)')
                print('PASS: production full-epoch training and accumulation match sample/update budgets, including the short final group (CPU, CUDA counters mocked)')
            if args.gpu:
                training.require_cuda()
                print("CUDA:", torch.cuda.get_device_name(0))
                short = replace(settings, start_row=0, row_limit=60, epochs=1, original_batch=2)
                with contextlib.redirect_stdout(io.StringIO()):
                    training.prepare_data(short, force=True)
                for technique in ("baseline", "mixed_precision", "gradient_checkpointing", "optimizer_8bit", "best_combination"):
                    result = training.run_training(short, technique, batch_size=2, epochs=1)
                    assert result["status"] == "success" and result["optimizer_steps"] > 0
                    if training.TECHNIQUES[technique]['amp']:
                        assert result['amp_dtype'] == 'torch.float16' and result['skipped_updates'] == 0
                        print('PASS CUDA: T4 FP16 autocast and GradScaler, with actual completed updates')
                    print("PASS CUDA training:", technique)
                result = training.run_training(short, "baseline", batch_size=2, epochs=1, accumulation_steps=2)
                assert result["optimizer_steps"] == 11
                print("PASS CUDA gradient accumulation")
                result = training.run_training(short, 'gradient_checkpointing', batch_size=128,
                                                epochs=1, memory_probe=True)
                assert result['measurement_kind'] == 'capacity_only_repeated_training_rows'
                assert result['observed_first_batch'] == result['samples_seen'] == 128
                assert result['optimizer_steps'] == 1 and 'test_accuracy' not in result
                print('PASS CUDA regression: oversized physical batch runs one actual update and never reports quality accuracy')
                # Tiny BERT tensors can be below bitsandbytes' 4096-element
                # threshold. Exercise a genuinely quantized optimizer state too.
                import bitsandbytes as bnb
                large_layer = torch.nn.Linear(256, 256).cuda()
                quantized_optimizer = bnb.optim.AdamW8bit(large_layer.parameters(), lr=1e-3)
                for _ in range(2):
                    quantized_optimizer.zero_grad(set_to_none=True)
                    large_layer(torch.randn(8, 256, device='cuda')).float().square().mean().backward()
                    quantized_optimizer.step()
                assert quantized_optimizer.state[large_layer.weight]['state1'].dtype == torch.uint8
                del large_layer, quantized_optimizer
                torch.cuda.empty_cache()
                print('PASS CUDA: actual 8-bit optimizer-state kernel on a large parameter tensor')
            if args.vllm:
                training.require_cuda()
                from vllm import LLM, SamplingParams
                llm = LLM(model=serving.DEFAULT_MODEL, dtype="float16", max_model_len=512,
                          gpu_memory_utilization=0.5, enforce_eager=True)
                outputs = llm.generate(["Explain GPU batching in one sentence."],
                                       SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)
                assert len(outputs[0].outputs[0].token_ids) > 0
                print("PASS: real vLLM model load and generation")
    print("SMOKE TEST PASSED. Synthetic test data was temporary and is not assignment evidence.")
    if not args.gpu:
        print("GPU tests NOT RUN. On a supported NVIDIA GPU, run: .venv/bin/python smoke_test.py --gpu --vllm")


if __name__ == "__main__":
    main()
