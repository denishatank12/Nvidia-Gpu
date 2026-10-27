# HW4.a saved BERT checkpoint

Download: [original Colab results archive](https://github.com/denishatank12/Nvidia-Gpu/releases/download/hw4-5/GPU_II_RESULTS.zip).

The checkpoint is stored as a GitHub release asset rather than a large ordinary Git blob. The archive includes all six saved model/tokenizer files under `gpu-ii/checkpoints/hw4a_bert_classifier/`.

Archive size: 439,949,104 bytes. SHA-256:

```text
68434bf0b59c1a49bf22a88b0ac34dd692387e493a2ef84d3fb899a66d940159
```

The original archive is immutable run evidence and contains the original reports. Use the corrected `HW4A_METRICS.md`, `METRICS.md`, and `AI_USE.md` in the tagged repository for the reviewed findings.

## Restore and reproduce

From a new empty directory, extract the checkpoint with:

```bash
unzip GPU_II_RESULTS.zip 'gpu-ii/checkpoints/hw4a_bert_classifier/*'
```

To reproduce training, use the repository's GPU setup and dataset preparation, then run this command from `gpu-ii/`:

```bash
.venv/bin/python hw4a_training.py --settings settings.json
```

This writes the classifier and tokenizer to `checkpoints/hw4a_bert_classifier/`, reloads them, and writes the two custom predictions to `outputs/hw4a_custom_inference.csv`. The completed run used a Tesla T4, three epochs, physical batch 16, and SEED 8834.

## Independent checkpoint check

After download, the six checkpoint files were extracted and loaded again on CPU using `BertForSequenceClassification.from_pretrained(..., local_files_only=True)` and the saved tokenizer. The two saved example sentences produced class IDs `[2, 0]`, corresponding to positive (`1.0`) and negative (`-1.0`), matching the recorded GPU run. This verifies reload and inference, not a second training run.

| File | Bytes | SHA-256 |
|---|---:|---|
| model.safetensors | 437961724 | `742f011561ba11ba9d0341979164ba27580efea95007cbeee3abbf76c08a4c11` |
| config.json | 819 | `3e6aa1670b197517c92a3021f95e131f2f85d40802e4cca8e29d31b941cac99b` |
| tokenizer.json | 711396 | `d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66` |
| tokenizer_config.json | 1221 | `1f92c93d2e31974acffab86d48e80a9357acb18c68346c5472ef1d12c72d5d77` |
| special_tokens_map.json | 125 | `b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3` |
| vocab.txt | 231508 | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` |
