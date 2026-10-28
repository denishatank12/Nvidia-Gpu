"""Integration test: explicitly synthetic fixtures, discarded after the test."""
import copy
import json
import tempfile
from pathlib import Path
import make_reports as reports


def main():
    with tempfile.TemporaryDirectory(prefix='gpu-ii-SYNTHETIC-test-') as temporary:
        root=Path(temporary);out=root/'outputs';out.mkdir()
        old_root,old_out=reports.ROOT,reports.OUT
        reports.ROOT,reports.OUT=root,out
        def put(name,value):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
        gpu={'name':'SYNTHETIC CPU FIXTURE, NOT GPU EVIDENCE','uuid':'GPU-SYNTHETIC',
             'memory_total_gb':6.,'driver_version':'fixture','torch_cuda':'fixture'}
        errors=[{'source_row':i,'text':f'bad fake promise {i}','true_class':'-1.0','predicted_class':'1.0'} for i in range(5)]
        base={'status':'success','technique':'baseline','physical_batch':16,'effective_batch':16,
              'accumulation_steps':1,'optimizer_steps':657,'samples_seen':10500,'epochs':3,
              'throughput_samples_per_second':50.,'peak_memory_allocated_gb':3.,
              'training_wall_seconds_including_validation':220.,'test_accuracy':.8,
              'best_validation_epoch':2,'worst_class_label':'-1.0','five_worst_class_mistakes':errors,
              'epoch_metrics':[{'epoch':i,'training_loss':1/i,'validation_loss':.8/i,
                                'validation_accuracy':v,'training_seconds':70.} for i,v in [(1,.7),(2,.82),(3,.81)]],
              'per_class':[{'class_label':str(i),'precision':.8,'recall':.8,'support':250} for i in [-1,0,1]]}
        settings={'original_batch':16,'epochs':3}
        manifest={'source_sha256':'SYNTHETIC-HASH','preparation_settings':settings,'requested_row_range_zero_based':[834,5833],
                  'split_sizes':{'train':3500,'validation':750,'test':750},'dataset_name':'SYNTHETIC ONLY'}
        original={'gpu':gpu,'training':base,'dataset_manifest':manifest,'reloaded_custom_inference':[
            {'text':'fixture good','predicted_class':'1.0','confidence':.9},{'text':'fixture bad','predicted_class':'-1.0','confidence':.9}]}
        put('outputs/hw4a_training_summary.json',original)
        names=['baseline','mixed_precision','gradient_checkpointing','optimizer_8bit','best_combination','gradient_accumulation']
        runs={}
        for i,name in enumerate(names):
            row=copy.deepcopy(base);row.update(technique=name,throughput_samples_per_second=[50,100,35,48,98,45][i],peak_memory_allocated_gb=[3,2.5,2,2.5,2.1,2.2][i])
            if name=='gradient_accumulation':row.update(physical_batch=4,accumulation_steps=4)
            runs[name]=row
        scaling=[]
        for name in names[:-1]:
            success=copy.deepcopy(runs[name]);success.update(physical_batch=16,observed_first_batch=16,mean_time_per_epoch_seconds=70.,measurement_kind='full_training_epoch')
            scaling.extend([success,{'technique':name,'physical_batch':32,'status':'oom'}])
        # Exercise the boundary where a memory-capacity batch exceeds all
        # available rows, but the quality run must not use repeated rows.
        scaling = [r for r in scaling if not (r['technique']=='gradient_checkpointing' and r['status']=='oom')]
        scaling.extend([{'technique':'gradient_checkpointing','physical_batch':8192,
                         'observed_first_batch':8192,'status':'success',
                         'measurement_kind':'capacity_only_repeated_training_rows'},
                        {'technique':'gradient_checkpointing','physical_batch':16384,'status':'oom'}])
        maxima={n:16 for n in names[:-1]};maxima['gradient_checkpointing']=8192
        put('outputs/training_summary.json',{'gpu':gpu,'settings':settings,'dataset_manifest':manifest,'full_runs':runs,
            'batch_scaling':scaling,'maximum_batches':maxima})
        put('outputs/controlled_training.json',{'runs':runs})
        serving={'model':'SYNTHETIC TEST MODEL','precision':'float16','max_model_len':4096,'prefix_caching':False,
                 'output_tokens':64,'gpu_memory_utilization':.7,
                 'latency_definition':'vLLM individual finished_time minus client submission time; synthetic fixture only.',
                 'rows':[{'configuration':n,'batch_size':b,'tokens_per_second':fps,'p50_latency_ms':lat,
                          'p95_latency_ms':lat+10,'requests_measured':100,'measured_batches':100 if b==1 else 20}
                         for n,b,fps,lat in [('HF pipeline, batch 1',1,30,2000),('vLLM, batch 1',1,60,1000),('vLLM, batch 8',8,400,1200),('vLLM, batch 32',32,1000,2500)]]}
        put('outputs/hw4a_serving_summary.json',serving);put('outputs/serving_summary.json',serving)
        put('outputs/context_summary.json',{'longest_verified_batch8_prompt':512,'first_failing_bracket':640,
            'boundary_resolution_tokens':128,'limit_kind':'cuda_oom','configuration_note':'SYNTHETIC TEST ONLY',
            'original_position_limit':32768,'records':[{'context_length':512,'status':'success',
            'all_eight_resident_together':True,'measured_active_kv_bytes':4096,'peak_running_requests':8},
            {'context_length':640,'status':'cuda_oom'}]})
        for name in ['model.safetensors','config.json','tokenizer.json']:
            p=root/'checkpoints/hw4a_bert_classifier'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('SYNTHETIC FIXTURE, NOT WEIGHTS')
        for name in ['gpu_uuid_before.txt','gpu_uuid_after.txt']:(out/name).write_text('GPU-SYNTHETIC')
        import matplotlib.pyplot as plt
        for name in ['hw4a_learning_curves.png','hw4a_confusion_matrix.png','batch_size_throughput.png']:
            fig,ax=plt.subplots();ax.plot([1,2],[2,1]);ax.set_title('SYNTHETIC TEST ONLY');fig.savefig(out/name);plt.close(fig)
        try:
            reports.main()
            assert json.loads((out/'verification.json').read_text())['all_passed']
            assert 'SYNTHETIC CPU FIXTURE' in (root/'METRICS.md').read_text()
            assert '| gradient_checkpointing | 8192 | 16 |' in (root/'METRICS.md').read_text()
            assert 'SYNTHETIC-HASH' in (root/'HW4A_METRICS.md').read_text()
            assert len(list(out.glob('*.png')))==4
            print('PASS: reports, plots, matched ranking, capacity bracket, and artifact verification on temporary synthetic fixtures')
        finally:reports.ROOT,reports.OUT=old_root,old_out
    print('REPORTING TEST PASSED. Fixtures deleted; no GPU results fabricated.')


if __name__=='__main__':main()
