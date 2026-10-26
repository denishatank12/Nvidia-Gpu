# HW4.a model checkpoint

The executed HW4.a notebook saves the fine-tuned classifier and tokenizer to:

`checkpoints/hw4a_bert_classifier/`

The checkpoint is intentionally excluded from ordinary Git tracking because it contains the full BERT weights. The exact command that creates and verifies it is:

```bash
.venv/bin/python hw4a_training.py --settings settings.json
```

The script immediately reloads the checkpoint and writes two predictions to `outputs/hw4a_custom_inference.csv`. Before final submission, place the large checkpoint in approved cloud storage or Git LFS and add its shareable link below.

Checkpoint link after the RTX 4090 run: **pending upload**
