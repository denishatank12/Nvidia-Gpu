#!/usr/bin/env python3
"""Read-only checks for the October 28 T4 submission. No training or report edits."""
from pathlib import Path
import csv
import json
import math
import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / name).read_text())

def run_audit():
    original = read('outputs/hw4a_training_summary.json')
    training = read('outputs/training_summary.json')
    controls = read('outputs/controlled_training.json')['runs']
    context = read('outputs/context_summary.json')
    checks = {}
    def check(name, condition):
        checks[name] = bool(condition)
    manifest = original['dataset_manifest']
    check('same source and split settings', manifest == training['dataset_manifest'])
    check('exact source rows', manifest['requested_row_range_zero_based'] == [834, 5833])
    check('fixed split', manifest['split_sizes'] == {'train':3500, 'validation':750, 'test':750})
    check('recorded actual T4', original['gpu'] == training['gpu'] and original['gpu']['name'] == 'Tesla T4')
    notebook = nbformat.read(ROOT / 'GPU_II_Colab_Run_All.ipynb', 4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == 'code']
    check('all ten cells executed in order', [c.execution_count for c in code] == list(range(1,11)))
    check('no saved error traceback', not any(o.output_type == 'error' for c in code for o in c.outputs))
    check('all six controls present', set(controls) == {'baseline','mixed_precision','gradient_checkpointing','gradient_accumulation','optimizer_8bit','best_combination'})
    check('same matched-update budget', all(all(r[k] == controls['baseline'][k] for k in ['effective_batch','optimizer_steps','samples_seen','epochs']) for r in controls.values()))
    runs = [original['training'], *training['full_runs'].values(), *controls.values()]
    check('actual full-epoch training budgets', all(r['epochs']==3 and r['samples_seen']==10500 and r['optimizer_steps']==3*math.ceil(3500/r['effective_batch']) and r['skipped_updates']==0 for r in runs))
    check('confusion-derived accuracies', all(sum(map(sum,r['confusion_matrix']))==750 and math.isclose(sum(r['confusion_matrix'][i][i] for i in range(3))/750,r['test_accuracy']) for r in runs))
    for technique, maximum in training['maximum_batches'].items():
        rows = [r for r in training['batch_scaling'] if r['technique']==technique]
        check('real memory wall '+technique, rows[-1]['status']=='oom' and rows[-1]['physical_batch']==2*maximum and all(r['observed_first_batch']==r['physical_batch'] for r in rows[:-1]))
    for prefix in ['hw4a_', '']:
        summary = read('outputs/'+prefix+'serving_summary.json')
        with (ROOT / 'outputs' / (prefix+'request_timings.csv')).open(newline='') as handle:
            raw = list(csv.DictReader(handle))
        for row in summary['rows']:
            records = [r for r in raw if r['configuration']==row['configuration']]
            values = [float(r['latency_ms']) for r in records]
            ok = len(records)==row['requests_measured'] and len(records)>=100
            ok &= math.isclose(float(np.percentile(values,50)),row['p50_latency_ms'])
            ok &= math.isclose(float(np.percentile(values,95)),row['p95_latency_ms'])
            ok &= sum(int(r['generated_tokens']) for r in records)==64*len(records)==row['generated_tokens']
            ok &= math.isclose(row['tokens_per_second'],row['generated_tokens']/row['measured_wall_seconds'])
            if row['configuration'].startswith('vLLM'):
                ok &= row['measured_batches']>=20 and all(math.isclose(float(r['latency_ms']),1000*(float(r['engine_finished_time'])-float(r['client_submission_time'])),abs_tol=1e-7) for r in records)
            check('raw serving '+(prefix or 'hw45_')+row['configuration'],ok)
    records = context['records']
    check('observed cache storage arithmetic', all(r['cache_pool_bytes']==r['cache_pool_blocks']*r['bytes_per_cache_block'] and r['measured_active_kv_bytes']==r['peak_used_cache_blocks']*r['bytes_per_cache_block'] for r in records))
    check('simultaneous context boundary', context['longest_verified_batch8_prompt']==29296 and context['first_failing_bracket']==29424 and context['boundary_resolution_tokens']==128 and context['limit_kind']=='batch8_admission_limit')
    check('both UUID captures', all(original['gpu']['uuid'] in (ROOT/'outputs'/n).read_text() for n in ['gpu_uuid_before.txt','gpu_uuid_after.txt']))
    ai = (ROOT/'AI_USE.md').read_text()
    check('reviewed AI-use text appears in notebook notes', ai.removeprefix('# AI-use appendix\n') in notebook.cells[-1].source)
    check('clarified throughput maximum', 'highest measured FP32 throughput was at batch 16' in (ROOT/'METRICS.md').read_text())
    check('checkpoint download documented', 'GPU_II_T4_RESULTS_6ec28b7ced2a.zip' in (ROOT/'MODEL_CHECKPOINT.md').read_text())
    check('Part C explicitly skipped', 'Part C skipped (optional)' in (ROOT/'METRICS.md').read_text())
    for name, passed in checks.items():
        print(('PASS: ' if passed else 'FAIL: ')+name)
    if not all(checks.values()):
        raise RuntimeError('Submission artifact audit failed; no files were changed.')
    print('READ-ONLY ARTIFACT AUDIT PASSED. No GPU experiment rerun or report regenerated.')
    return checks

if __name__ == '__main__':
    run_audit()
