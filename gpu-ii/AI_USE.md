# AI-use appendix

## What the assistant helped with

I used an AI assistant to write and organize the BERT training code, GPU memory experiments, serving benchmark, and reporting scripts. It also helped interpret the results and check the submission files. I ran the combined notebook in Google Colab and downloaded the executed notebook and results archive. I am not claiming that I wrote the generated scripts without help.

## A specific mistake

The generated report still referred to the GPU I originally planned to use, although the actual run used a Tesla T4. One incorrect sentence was:

```text
I would apply mixed precision first because the 4090 has fast tensor cores and AMP normally improves both memory use and speed.
```

The original ranking also included the combined configuration instead of gradient accumulation. These were reporting mistakes, not experimental findings.

## How the mistakes were found

After the run, I asked the assistant to audit the downloaded notebook and results archive. The recorded GPU name and UUID showed a Tesla T4. Comparing the ranking with the four requested techniques exposed the missing accumulation row. The maximum-batch runs also had different optimizer-update counts, so their accuracy differences could not be attributed to the techniques alone.

## What changed and what was checked

The reporting code now reads the GPU name from the measured summary. The corrected ranking includes all four techniques and names the matched reference for each comparison. The conclusions distinguish maximum-batch experiments from common-batch probes and explain that no context-length GPU-memory failure occurred before the configured model limit. Individual error diagnoses discuss the actual tweets, including possible label noise, without claiming to know the model's internal reasoning.

The assistant checked that the raw training and serving summaries were unchanged and that the revised CSVs changed diagnoses, not predictions or labels. It also reloaded the saved BERT checkpoint on CPU and reproduced the two predicted classes. There is one submission notebook: its original code cells, execution counts, and outputs remain intact, and clearly labelled post-run Markdown notes contain the corrected analysis. The separate reporting scripts were corrected as well. These checks do not mean that a new GPU benchmark was run after the edits.
