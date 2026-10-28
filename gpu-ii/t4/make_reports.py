"""Create findings from this run only. No prior GPU scores are inserted."""
from __future__ import annotations
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from hw45_serving import describe_latency_tradeoff

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'


def read(name):
    return json.loads((ROOT / name).read_text())


def save(name, lines):
    (ROOT / name).write_text('\n'.join(lines) + '\n')


def table(rows, fields, labels=None):
    labels = labels or fields
    lines = ['| ' + ' | '.join(labels) + ' |', '|' + '|'.join(['---'] * len(fields)) + '|']
    for row in rows:
        values = []
        for key in fields:
            value = row.get(key)
            if isinstance(value, float): value = f'{value:.4f}'
            values.append(str(value if value is not None else 'not measured').replace('|', '\\|').replace('\n', ' '))
        lines.append('| ' + ' | '.join(values) + ' |')
    return lines


def diagnosis(row):
    text = ' '.join(row['text'].split())
    words = set(text.lower().split())
    negative = sorted(words & {'bad', 'hate', 'fake', 'failure', 'looting', 'hypocrite', 'wasted', 'finished', 'corrupt', 'chor', 'bullshit', 'crookcheap', 'trolls', 'wrong', 'lies'})
    positive = sorted(words & {'good', 'great', 'love', 'hope', 'praise', 'respect', 'leadership', 'support', 'hardworking', 'better', 'best', 'happy'})
    if negative and positive:
        return f"The text mixes negative wording ({', '.join(negative[:3])}) with positive wording ({', '.join(positive[:3])}); a single sentiment label can hide the different targets or clauses."
    if negative:
        return f"The wording includes {', '.join(negative[:3])}, but the prediction is {row['predicted_class']}; those critical cues are not reflected in the output. This is a text-based diagnosis, not proof of the model's internal cause."
    if positive:
        return f"The wording includes {', '.join(positive[:3])}; that may explain the prediction {row['predicted_class']}, while the recorded label is {row['true_class']}. Missing context or label noise should be considered before assuming the annotation is unambiguous."
    return f"The phrase '{text[:100]}' has little explicit sentiment wording; context or a target-specific interpretation may account for the disagreement between label {row['true_class']} and prediction {row['predicted_class']}."


def ranking(controls):
    baseline = controls['baseline']
    rows = []
    for name in ['mixed_precision', 'gradient_checkpointing', 'gradient_accumulation', 'optimizer_8bit']:
        row = controls[name]
        assert all(row[k] == baseline[k] for k in ['epochs', 'samples_seen', 'optimizer_steps', 'effective_batch'])
        saved = baseline['peak_memory_allocated_gb'] - row['peak_memory_allocated_gb']
        lost = baseline['throughput_samples_per_second'] - row['throughput_samples_per_second']
        score = saved / lost if saved > 0 and lost > 0 else None
        explanation = 'no memory saving' if saved <= 0 else ('no throughput loss' if lost <= 0 else f'{score:.4f}')
        rows.append({'technique': name, 'memory_saved_GiB': saved, 'throughput_change': -lost,
                     'GiB_saved_per_sample_per_second_lost': explanation,
                     '_rank': (2, saved) if saved > 0 and lost <= 0 else ((1, score) if score is not None else (0, saved))})
    rows.sort(key=lambda r: r['_rank'], reverse=True)
    for i, r in enumerate(rows, 1): r['rank'] = i; r.pop('_rank')
    return rows


def verification():
    original = read('outputs/hw4a_training_summary.json')
    training = read('outputs/training_summary.json')
    controlled = read('outputs/controlled_training.json')
    first, last = read('outputs/hw4a_serving_summary.json'), read('outputs/serving_summary.json')
    context = read('outputs/context_summary.json')
    base = controlled['runs']['baseline']
    checks = {
        'same dataset hash': original['dataset_manifest']['source_sha256'] == training['dataset_manifest']['source_sha256'],
        'same preparation settings': original['dataset_manifest']['preparation_settings'] == training['dataset_manifest']['preparation_settings'],
        'exact source rows': original['dataset_manifest']['requested_row_range_zero_based'] == [834, 5833],
        'fixed split': original['dataset_manifest']['split_sizes'] == {'train': 3500, 'validation': 750, 'test': 750},
        'all independent and combined memory sweeps reached OOM': all(any(r['technique'] == name and r['status'] == 'oom' for r in training['batch_scaling']) for name in training['maximum_batches']),
        'successful probes materialized their requested batch': all(r.get('observed_first_batch') == r['physical_batch'] for r in training['batch_scaling'] if r['status'] == 'success'),
        'quality runs never repeat rows to inflate a physical batch': all(r['physical_batch'] <= manifest_size for r in training['full_runs'].values() for manifest_size in [training['dataset_manifest']['split_sizes']['train']]),
        'six matched training conditions': len(controlled['runs']) == 6,
        'same controlled effective batch and training updates': all(all(r[k] == base[k] for k in ['effective_batch', 'epochs', 'samples_seen', 'optimizer_steps']) for r in controlled['runs'].values()),
        'saved BERT reloaded': len(original['reloaded_custom_inference']) == 2,
        'five original worst-class mistakes': len(original['training']['five_worst_class_mistakes']) == 5,
        'baseline and repeated serving configurations match': all(first[k] == last[k] for k in ['model', 'precision', 'max_model_len', 'prefix_caching', 'output_tokens', 'gpu_memory_utilization']),
        'serving batches recorded in both stages': all([r['batch_size'] for r in s['rows']] == [1, 1, 8, 32] for s in [first, last]),
        'each serving row has at least 100 real request samples': all(r['requests_measured'] >= 100 for s in [first, last] for r in s['rows']),
        'at least 20 measured vLLM batches per shape': all(r['measured_batches'] >= 20 for s in [first, last] for r in s['rows'][1:]),
        'request timestamps, not batch copies': all('individual finished_time minus client submission time' in s['latency_definition'] for s in [first, last]),
        'eight concurrent KV allocations observed': any(r['status'] == 'success' and r['all_eight_resident_together'] and r['measured_active_kv_bytes'] > 0 for r in context['records']),
        'capacity boundary or unmodified model limit identified': context['limit_kind'] in ['model_position_limit', 'batch8_admission_limit', 'cuda_oom', 'kv_cache_capacity'],
        'checkpoint weights and tokenizer exist': all((ROOT / 'checkpoints/hw4a_bert_classifier' / name).exists() for name in ['model.safetensors', 'config.json', 'tokenizer.json']),
        'both GPU UUID captures exist': all((OUT / name).exists() for name in ['gpu_uuid_before.txt', 'gpu_uuid_after.txt']),
        'all required figures exist': all((OUT / name).exists() for name in ['hw4a_learning_curves.png', 'hw4a_confusion_matrix.png', 'batch_size_throughput.png', 'controlled_losses.png']),
        'optional Part C skip': 'Part C skipped (optional)' in (ROOT / 'METRICS.md').read_text(),
    }
    result = {'checks': checks, 'all_passed': all(checks.values()),
              'scope': 'Checks measured artifacts only. Publishing, private-repository access, a final tag, and student review happen after this run.'}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    for k, v in checks.items(): print(('PASS: ' if v else 'FAIL: ') + k)
    if not result['all_passed']: raise RuntimeError('A measured-artifact check failed. Keep the logs and do not call this run complete.')
    return result


def main():
    original = read('outputs/hw4a_training_summary.json')
    training = read('outputs/training_summary.json')
    controls = read('outputs/controlled_training.json')['runs']
    serving, prior = read('outputs/serving_summary.json'), read('outputs/hw4a_serving_summary.json')
    context = read('outputs/context_summary.json')
    baseline, gpu, manifest = original['training'], original['gpu'], original['dataset_manifest']
    intro = ['Student: Denisha Ketan Tank', '',
             'SID4 = 8834, SEED = 8834, SLICE = 834, HP_ID = 2, CLS_A = 4, CLS_B = 6.', '',
             f"Actual GPU: {gpu['name']}; UUID: `{gpu['uuid']}`; VRAM: {gpu['memory_total_gb']:.2f} GiB; driver: {gpu['driver_version']}; PyTorch CUDA: {gpu['torch_cuda']}.", '',
             f"Dataset: Kaggle saurabhshahane/twitter-sentiment-dataset, {manifest['dataset_name']}. Zero-based rows 834 through 5833 inclusive, Python slice [834:5834). SHA-256: `{manifest['source_sha256']}`. The fixed SEED 8834 stratified split is 3,500/750/750. Every comparison reuses these exact saved split indices.", '',
             'Memory is reported in binary GiB. Peak allocated training memory, total GPU-process usage, reserved cache storage, and actively occupied cache blocks are different measurements.', '']
    intro += ['Training parameters remain FP32. AMP uses T4-compatible FP16 with conservative loss scaling; the actual autocast dtype is saved in each result. T4 does not support native BF16 or TF32. A recoverable overflow repeats the same logical update at lower scale, restores dropout RNG, and changes no weights until successful. The repeat cost stays in measured time and memory, and overflow retry counts are saved in the raw results. Persistent numerical failure stops instead of being called a successful update.', '']
    intro += ['Accuracy columns in the tables are fractions from 0 to 1. Dataset labels -1, 0, and 1 mean negative, neutral, and positive sentiment.', '']
    epochs = baseline['epoch_metrics']
    best_epoch = baseline['best_validation_epoch']
    waste = sum(r['training_seconds'] for r in epochs if r['epoch'] > best_epoch)
    h4a = ['# HW4.a: BERT sentiment and serving', '', *intro, '## BERT results', '',
           f"The original model used batch {baseline['physical_batch']}, sequence length 128, learning rate 0.00002, and three full epochs. Test accuracy was {100 * baseline['test_accuracy']:.2f}%. Training plus validation took {baseline['training_wall_seconds_including_validation']:.2f} seconds, with {baseline['peak_memory_allocated_gb']:.3f} GiB peak training allocation.", '',
           f"The best validation accuracy occurred at epoch {best_epoch}. " + (f"Training after that epoch cost {waste:.2f} seconds without improving the best validation accuracy." if waste else 'The best point was the final epoch, so no later training cost is claimed.'), '',
           *table(epochs, ['epoch', 'training_loss', 'validation_loss', 'validation_accuracy', 'training_seconds']), '',
           '![Training and validation curves](outputs/hw4a_learning_curves.png)', '',
           *table(baseline['per_class'], ['class_label', 'precision', 'recall', 'support']), '',
           '![Confusion matrix](outputs/hw4a_confusion_matrix.png)', '',
           f"The lowest-recall class was {baseline['worst_class_label']}. The following are actual errors from that class. The diagnoses discuss wording, not proven internal model mechanisms.", '']
    for prefix, result, document in [('hw4a', baseline, h4a), ('baseline', training['full_runs']['baseline'], None)]:
        mistakes = [dict(r, diagnosis=diagnosis(r)) for r in result['five_worst_class_mistakes']]
        pd.DataFrame(mistakes).to_csv(OUT / (prefix + '_misclassified_examples.csv'), index=False)
        if document is not None: document.extend(table(mistakes, ['source_row', 'true_class', 'predicted_class', 'text', 'diagnosis']))
    h4a += ['', '## Saved model and custom inference', '',
            'The classifier and tokenizer were saved and reloaded before these two predictions.', '',
            *table(original['reloaded_custom_inference'], ['text', 'predicted_class', 'confidence']), '',
            '## HF and vLLM serving', '',
            prior['latency_definition'], '',
            'Each mode generates exactly 64 new tokens, is warmed up, and uses the same raw prompt family and FP16 model. Each row has at least 100 real requests. vLLM batches have at least 20 measured repetitions. Individual request timestamps and raw observations are saved, so requests are not all assigned the enclosing batch time.', '',
            *table(prior['rows'], ['configuration', 'tokens_per_second', 'p50_latency_ms', 'p95_latency_ms', 'requests_measured', 'measured_batches']), '',
            describe_latency_tradeoff(prior['rows']), '', '### My prompts and actual generated answers', '',
            *table(prior.get('examples', []), ['configuration', 'prompt', 'answer']), '']
    save('HW4A_METRICS.md', h4a)
    full_rows = [dict(r, technique=name) for name, r in training['full_runs'].items()]
    control_rows = [dict(r, technique=name) for name, r in controls.items()]
    fields = ['technique', 'physical_batch', 'effective_batch', 'optimizer_steps', 'throughput_samples_per_second', 'peak_memory_allocated_gb', 'test_accuracy']
    ranks = ranking(controls)
    pd.DataFrame(ranks).to_csv(OUT / 'technique_ranking.csv', index=False)
    pd.DataFrame([{k: r[k] for k in fields} for r in control_rows]).to_csv(OUT / 'controlled_training.csv', index=False)
    good = sorted([r for r in training['batch_scaling'] if r['technique'] == 'baseline' and r['status'] == 'success' and r.get('measurement_kind') != 'capacity_only_repeated_training_rows'], key=lambda r:r['physical_batch'])
    plateau = next((b['physical_batch'] for a, b in zip(good, good[1:]) if b['throughput_samples_per_second'] / a['throughput_samples_per_second'] - 1 < .05), good[-1]['physical_batch'])
    matched_base = controls['baseline']
    eligible = {k:v for k,v in controls.items() if v['epoch_metrics'][-1]['validation_accuracy'] >= matched_base['epoch_metrics'][-1]['validation_accuracy'] - .01}
    selected_name, selected = max(eligible.items(), key=lambda kv:kv[1]['throughput_samples_per_second'])
    metrics = ['# HW4.5: throughput, memory, and serving', '', *intro,
               '## Part A: batch scaling', '',
               'The clarification permits one GPU, so no second-card result or VRAM ratio is invented. Probes up to the training-set size are complete seeded one-epoch runs. Larger probes repeat training rows for one full physical-batch optimizer update solely to find the memory wall. They are explicitly labelled capacity-only, not used for accuracy or the epoch-throughput plateau. A fresh process is used after every attempt, including the expected OOM.', '',
               *table([r for r in training['batch_scaling'] if r['technique']=='baseline'], ['physical_batch', 'status', 'measurement_kind', 'throughput_samples_per_second', 'peak_memory_allocated_gb', 'mean_time_per_epoch_seconds']), '',
               f"The largest successful FP32 batch was {training['maximum_batches']['baseline']}. The first incremental throughput gain below 5% occurred at batch {plateau}; this is the stated plateau rule.", '',
               '![Batch scaling](outputs/batch_size_throughput.png)', '',
               '## Part B: maximum-batch experiments', '', *table(full_rows, fields), '',
               'Each quality configuration was trained for three full epochs at the largest successful power-of-two batch that does not exceed the original training split. Larger capacity-only memory limits are listed below, not claimed as quality-training batch sizes. Changing batch size changes optimizer-update counts, so these accuracy differences cannot be assigned solely to a technique.', '',
               *table([{'technique': n, 'memory_capacity_batch': b, 'quality_training_batch': training['full_runs'][n]['physical_batch']} for n,b in training['maximum_batches'].items()], ['technique', 'memory_capacity_batch', 'quality_training_batch']), '',
               '## Part B: matched-budget quality and memory controls', '', *table(control_rows, fields), '',
               f"All six controls use effective batch {matched_base['effective_batch']}, {matched_base['optimizer_steps']} optimizer updates, {matched_base['samples_seen']} examples, and three epochs. Accumulation intentionally uses a smaller physical batch with the same effective batch. Data order, initialization seed, learning rate, and examples are fixed. Dropout and floating-point arithmetic can still produce differences, so a single run does not establish statistical significance.", '',
               '![Controlled losses](outputs/controlled_losses.png)', '',
               '### Four independent techniques ranked by memory saved per throughput lost', '',
               *table(ranks, ['rank', 'technique', 'memory_saved_GiB', 'throughput_change', 'GiB_saved_per_sample_per_second_lost']), '',
               'The combined candidate is excluded from the independent ranking. A speedup with a memory saving has no positive throughput-loss denominator; a technique that saves no memory is labelled accordingly.', '',
               f"Checkpointing changed matched-batch throughput by {100 * (controls['gradient_checkpointing']['throughput_samples_per_second'] / matched_base['throughput_samples_per_second'] - 1):+.1f}%. The extra backward recomputation explains why memory savings may have a speed cost; the measured sign is reported even if a short run is noisy.", '',
               f"Matched accumulation used physical batch {controls['gradient_accumulation']['physical_batch']} times {controls['gradient_accumulation']['accumulation_steps']} accumulation steps. Its time to the same {matched_base['optimizer_steps']} updates was {controls['gradient_accumulation']['training_wall_seconds_including_validation']:.2f} seconds, versus {matched_base['training_wall_seconds_including_validation']:.2f} seconds for the direct batch.", '',
               f"I would first consider {ranks[0]['technique']} from this ranking, while checking its measured validation accuracy. On a 32 GiB card I would not add checkpointing if the required batch already fits and the measured recomputation cost is not worthwhile. That 32 GiB statement is a recommendation, not a second-GPU experiment.", '',
               '## Part C', '', 'Part C skipped (optional)', '',
               '## Part D: serving repeated on the same GPU', '', serving['latency_definition'], '',
               *table(serving['rows'], ['configuration', 'tokens_per_second', 'p50_latency_ms', 'p95_latency_ms', 'requests_measured', 'measured_batches']), '',
               describe_latency_tradeoff(serving['rows']), '',
               'The request samples within a batch can be correlated. More measured requests improve quantile resolution, but these results are still a benchmark under this workload, not a production p95 guarantee.', '']
    comparisons = []
    for a, b in zip(prior['rows'], serving['rows']):
        comparisons.append({'configuration': b['configuration'], 'HW4a_tokens_s': a['tokens_per_second'], 'HW45_tokens_s': b['tokens_per_second'],
                            'HW4a_p50_ms': a['p50_latency_ms'], 'HW45_p50_ms': b['p50_latency_ms'], 'HW4a_p95_ms': a['p95_latency_ms'], 'HW45_p95_ms': b['p95_latency_ms']})
    metrics += table(comparisons, list(comparisons[0])) + ['',
                f"Both stages used the same GPU and weights, FP16, 64 output tokens, individual-request timing, and frozen GPU-memory budget ({serving['gpu_memory_utilization']:.4f} of total VRAM). Background load and GPU clock/temperature may contribute to timing changes; no hardware upgrade is claimed. The actual cache-pool allocation is saved separately in each serving summary.", '',
                '### Observed KV-cache usage and simultaneous batch-8 capacity', '',
                'Cache-pool bytes come from the actual GPU tensor storages. Peak occupied blocks are observed after scheduler allocation and before model execution. Multiplying the observed occupied blocks by measured bytes per block gives the active reserved KV storage, including block rounding. This is not an isolated before/after nvidia-smi estimate. Prefix caching and chunked prefill are disabled for the capacity probes, and success requires all eight full prompts to reside together.', '',
                *table(context['records'], ['context_length', 'status', 'cache_pool_bytes', 'peak_used_cache_blocks', 'measured_active_kv_bytes', 'peak_running_requests']), '',
                f"The longest verified simultaneous batch-8 prompt was {context['longest_verified_batch8_prompt']} tokens per request. The limiting category was {context['limit_kind']}. " + (f"The successful/failing bracket ends at {context['first_failing_bracket']} and is at most {context['boundary_resolution_tokens']} tokens wide; an exact one-token boundary is not claimed." if context['first_failing_bracket'] is not None else f"Every attempted prompt fit through the unmodified model's {context['original_position_limit']}-token limit minus one output token. No out-of-range context is invented to force OOM."), '',
                context['configuration_note'], '',
                'CUDA OOM, an engine cache-allocation limit, and inability to admit eight full contexts are recorded separately. A non-memory exception stops the run rather than being relabelled as an OOM.', '',
                '## Part E: configuration and GPU-hours', '',
                f"My validation-based choice among the matched controls is {selected_name}. I selected the highest-throughput option whose final validation accuracy was within one percentage point of the matched FP32 baseline, rather than tuning on test accuracy. Its measured training allocation was {selected['peak_memory_allocated_gb']:.3f} GiB, throughput {selected['throughput_samples_per_second']:.2f} samples/s, wall time {selected['training_wall_seconds_including_validation']:.2f} seconds, and test accuracy {100 * selected['test_accuracy']:.2f}%.", '',
                f"One full fine-tune under that measured configuration used 1 GPU × {selected['training_wall_seconds_including_validation']:.2f} seconds / 3600 = {selected['training_wall_seconds_including_validation']/3600:.4f} GPU-hours. This excludes setup and the other experimental conditions.", '',
                '### HW4.5 summary', '',
                f"Selected matched-budget configuration: {selected_name}; {selected['peak_memory_allocated_gb']:.3f} GiB peak, {selected['throughput_samples_per_second']:.2f} samples/s, {selected['training_wall_seconds_including_validation']:.2f} seconds. It is an HW4.5 result, not an HW5 baseline.", '']
    save('METRICS.md', metrics)
    save('GPU_HOURS.md', ['# GPU-hour record', '', f"Actual GPU: {gpu['name']} ({gpu['uuid']}).", '',
         f"Chosen configuration: {selected_name}. One GPU × {selected['training_wall_seconds_including_validation']:.6f} seconds / 3600 = {selected['training_wall_seconds_including_validation']/3600:.6f} GPU-hours.", '',
         f"Original HW4.a baseline: {baseline['training_wall_seconds_including_validation']/3600:.6f} GPU-hours.", '',
         'These are training-plus-validation costs for individual three-epoch runs, not total allocation time for all experiments.'])
    # The archived checkpoint is complete even though no cloud upload or tag is invented.
    save('MODEL_CHECKPOINT.md', ['# BERT checkpoint', '', 'The results archive includes `checkpoints/hw4a_bert_classifier/`, containing the full classifier and tokenizer.', '',
         'Training command from this run directory: `./.venv/bin/python hw4a_training.py --settings settings.json`.', '',
         'The model was reloaded and tested during HW4.a. Before repository submission, publish this archive or checkpoint to accessible storage and include its link. The notebook does not silently publish data or create a GitHub tag.'])
    save('AI_USE.md', ['# AI-use appendix', '', 'I used an AI assistant for the experimental code, notebook organization, smoke tests, and result-report generation. The measurements in this folder come from my T4 run, not from an estimated GPU score.', '',
         'A specific error in the earlier benchmark was this latency calculation:', '', '```python', 'latencies_ms.extend([elapsed * 1000] * batch_size)', '```', '',
         'This assigns a single batch duration to every request and does not measure each request finishing independently. The assistant found this during a source-code audit. The new benchmark uses each request\'s own finished_time relative to client submission, also saves its arrival_time, and stores the raw observations. It warms each batch shape and measures more repetitions.', '',
         'The earlier quality comparison also changed batch size and optimizer-update count at once. The new run keeps a separate three-epoch, matched-effective-batch comparison for all four techniques and the combined candidate. Tests check that update counts and sample counts actually match.', '',
         'The assistant also replaced theoretical-only active KV estimates with actual cache tensor-storage and scheduler-block observations. Non-capacity errors are not hidden as OOM. I still need to understand and review the generated findings before submitting them as my work.'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, row in controls.items():
        x = [r['epoch'] for r in row['epoch_metrics']]
        axes[0].plot(x, [r['training_loss'] for r in row['epoch_metrics']], marker='o', label=name)
        axes[1].plot(x, [r['validation_loss'] for r in row['epoch_metrics']], marker='o', label=name)
    for ax, title in zip(axes, ['Matched training loss', 'Matched validation loss']):
        ax.set(title=title, xlabel='Epoch', ylabel='Cross-entropy'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'controlled_losses.png', dpi=170); plt.close(fig)
    verification()
    print('REPORTS AND MEASURED-ARTIFACT CHECKS COMPLETE')


if __name__ == '__main__': main()
