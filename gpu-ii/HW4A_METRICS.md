# HW4.a: BERT Fine-Tuning and vLLM Inference

Student: Denisha Ketan Tank

## Reproducibility and data

Personal parameters: SID4 = 8834, SEED = 8834, SLICE = 834, HP_ID = 2, CLS_A = 4, CLS_B = 6.

I used `Twitter_Data.csv` with SHA-256 `5a7af1ec5fc749ec8e5ea13c574aeb5c06254aa1c081e3421868079d5356b3f4`. The exact zero-based source rows were 834 through 5833 (Python slice `[834:5834)`). After recording missing rows, the fixed 70/15/15 split contained 3500 training, 750 validation, and 750 test examples. Every split and model initialization used SEED 8834.

## BERT fine-tuning

Model: `bert-base-uncased`. Batch size: 16. Epochs: 3. Peak allocated GPU memory: 2.467 GB. Training wall time including validation: 244.93 seconds. Test accuracy: 83.60%.

The best validation accuracy occurred at the final epoch, epoch 3, so this run did not continue past the best point.

| Epoch | Training loss | Validation loss | Validation accuracy | Training time (s) |
|---:|---:|---:|---:|---:|
| 1 | 0.9082 | 0.6851 | 71.60% | 73.26 |
| 2 | 0.5244 | 0.5202 | 80.80% | 77.30 |
| 3 | 0.2899 | 0.5094 | 82.40% | 77.27 |

### Per-class behavior

| Class | Precision | Recall | Support |
|---|---:|---:|---:|
| -1.0 | 0.7151 | 0.7665 | 167 |
| 0.0 | 0.8294 | 0.9000 | 270 |
| 1.0 | 0.9209 | 0.8179 | 313 |

The worst class by recall was **-1.0**. The confusion matrix is in `outputs/hw4a_confusion_matrix.png`.

### Five mistakes from the worst class

| Source row | True | Predicted | Tweet | Diagnosis |
|---:|---|---|---|---|
| 1667 | -1.0 | 1.0 | ratan naval tata family patriarch the billion tata empire has lavished praise prime minister narendra modi underscoring that his leadership what the country needsat this point time and urging citizens rally round his vision create new india | The text explicitly praises Modi and his leadership, so the positive prediction fits the visible wording; the negative dataset label may be noisy, although the missing original context prevents certainty. |
| 3127 | -1.0 | 1.0 | from rahul gandhis fake promise today from where will bring money this much money india than according him modi government failure and chowkidar chor than more than this money was india before five years than why not thought this | The tweet mixes criticism of a fake promise with a rhetorical comparison between political leaders; the positive prediction misses the critical tone, but the text alone does not establish which phrase drove it. |
| 4326 | -1.0 | 0.0 | talk about looting hypocrite modi took white money converted black gave trolls everone knows who else | Words such as looting, hypocrite, and trolls convey criticism; the neutral prediction misses these negative cues in the informal wording. |
| 5127 | -1.0 | 0.0 | took you hrs raju finally break silence this did you spend them thinking how you will bring modi into this without appearing monkey balancing that case you wasted that time | The writer mocks the other person with appearing monkey and wasted that time; treating this as neutral loses the criticism expressed through the rhetorical question. |
| 4162 | -1.0 | 1.0 | bullshit congressmen india controlled remote whose controller italian converted parsi claiming herself citizen the countrywhich only documents not her heart her name sonia ghandyif the hindusmuslimsetcwant goodthen vote only modi | The tweet attacks Congress but ends by supporting Modi, so it contains different sentiments toward different targets; the positive output may follow the ending rather than the overall negative label. |

The trained classifier was saved, reloaded from disk, and used on two custom sentences. The outputs are in `outputs/hw4a_custom_inference.csv`.

## vLLM serving benchmark

I used the different Hugging Face model `Qwen/Qwen2.5-0.5B-Instruct` with my own prompts. All configurations generated at most 64 tokens per request.

| Configuration | Tokens/sec | p50 latency (ms) | p95 latency (ms) |
|---|---:|---:|---:|
| HF pipeline, batch 1 | 28.53 | 2114.15 | 2823.26 |
| vLLM, batch 1 | 66.08 | 949.92 | 1037.75 |
| vLLM, batch 8 | 478.11 | 1066.14 | 1107.22 |
| vLLM, batch 32 | 1631.81 | 1249.70 | 1270.21 |

None of the tested configurations showed worse p50 latency than the HF baseline while also increasing throughput over the preceding configuration. Batching can increase total work completed per second while individual requests wait longer. For an interactive application, p50 and especially p95 latency matter because a user waits for a response. For an offline job, aggregate tokens per second is usually more important.
