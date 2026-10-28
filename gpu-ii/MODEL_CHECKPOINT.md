# HW4.a saved BERT checkpoint

[Download the October 28 T4 results archive](https://github.com/denishatank12/Nvidia-Gpu/releases/download/hw4-5/GPU_II_T4_RESULTS_6ec28b7ced2a.zip).

The archive contains all six classifier/tokenizer files under `gpu-ii/checkpoints/hw4a_bert_classifier/`. It is stored as a release asset rather than a large ordinary Git blob.

Archive size: 418,144,191 bytes. SHA-256:

```text
a3c4d4c7ed20163b82564410e418cd053381046911811daf34f816e203471b8a
```

This is the exact archive supplied after the successful run; it was not repacked to make the reports appear rerun. The tagged repository contains the reviewed AI-use text and clarified FP32 plateau wording.

## Restore and reproduce

Extract the checkpoint from the downloaded archive:

```bash
unzip GPU_II_T4_RESULTS_6ec28b7ced2a.zip 'gpu-ii/checkpoints/hw4a_bert_classifier/*'
```

The exact training command, from the original notebook's run directory, is:

```bash
./.venv/bin/python hw4a_training.py --settings settings.json
```

Use the self-contained T4 notebook for complete environment and dataset setup. The recorded run uses one Tesla T4, three epochs, physical batch 16 and SEED 8834.

## Independent checkpoint verification

During the audit the archived classifier and tokenizer were loaded offline on CPU. Both custom predictions and their reported confidence reproduced within tolerance. All five documented misclassifications also reproduced. This verifies saved-model inference, not a second training run.

| File | Bytes | SHA-256 |
|---|---:|---|
| model.safetensors | 437961724 | `742f011561ba11ba9d0341979164ba27580efea95007cbeee3abbf76c08a4c11` |
| config.json | 819 | `3e6aa1670b197517c92a3021f95e131f2f85d40802e4cca8e29d31b941cac99b` |
| tokenizer.json | 711396 | `d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66` |
| tokenizer_config.json | 1221 | `1f92c93d2e31974acffab86d48e80a9357acb18c68346c5472ef1d12c72d5d77` |
| special_tokens_map.json | 125 | `b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3` |
| vocab.txt | 231508 | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` |

The original executed notebook is also preserved as [a release asset](https://github.com/denishatank12/Nvidia-Gpu/releases/download/hw4-5/GPU_II_T4_Run_All_original_executed.ipynb). The older `GPU_II_RESULTS.zip` asset belongs to the previous run.
