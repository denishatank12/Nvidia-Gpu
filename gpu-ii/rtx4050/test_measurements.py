"""Offline measurement tests plus an optional real-GPU vLLM interface check."""
import argparse
import math
from types import SimpleNamespace as NS
import numpy as np
import torch
import accurate_serving as serving


def main(gpu=False):
    output = NS(metrics=NS(arrival_time=10.0, first_token_time=10.1, finished_time=10.4))
    latency, first = serving.request_timing(output)
    assert math.isclose(latency, 400) and math.isclose(first, 100)
    assert math.isclose(serving.request_timing(output, 9.9)[0], 500)
    try:
        serving.request_timing(NS(metrics=None))
    except (RuntimeError, AttributeError): pass
    else: raise AssertionError('Missing timestamps must not be replaced by batch latency')
    rows = [{'latency_ms': value, 'repeat': i // 2} for i, value in enumerate([1, 2, 9, 10])]
    summary = serving.summarize('test', 2, rows, 2, 128)
    assert summary['tokens_per_second'] == 64 and summary['measured_batches'] == 2
    assert summary['p50_latency_ms'] == np.percentile([1, 2, 9, 10], 50)
    assert serving.capacity_error('ValueError: unrelated loader failure') is None
    assert serving.capacity_error('CUDA out of memory') == 'cuda_oom'
    assert serving.capacity_error('No available memory for the cache blocks') == 'kv_cache_capacity'
    tensors = [torch.zeros(2, 16, 4, dtype=torch.float16) for _ in range(3)]
    cache = NS(gpu_cache=tensors, num_gpu_blocks=16, block_size=4)
    scheduler = NS()
    fake = NS(llm_engine=NS(model_executor=NS(driver_worker=NS(worker=NS(cache_engine=[cache]))), scheduler=[scheduler]))
    _, pool, blocks, bytes_per_block, block_size = serving.cache_details(fake)
    assert pool == sum(t.numel() * t.element_size() for t in tensors)
    assert blocks == 16 and bytes_per_block == pool // 16 and block_size == 4
    print('PASS: real per-request arithmetic, percentiles, cache storage accounting, and capacity-error classification')
    if gpu:
        from vllm import SamplingParams
        llm = serving.make_llm(serving.memory_utilization(), context=512, token_budget=512)
        for batch in (1, 8, 32):
            outputs = llm.generate(['Explain GPU batching.'] * batch,
                                   SamplingParams(temperature=0, max_tokens=8, ignore_eos=True), use_tqdm=False)
            assert len(outputs) == batch
            for result in outputs:
                assert serving.request_timing(result)[0] > 0
            print('PASS: real vLLM request timestamps at batch', batch)
        scheduler, pool, blocks, _, _ = serving.cache_details(llm)
        assert pool > 0 and blocks > 0
        print('PASS: real GPU KV-cache tensors and scheduler interface')
        observed = []
        original_schedule = scheduler.schedule
        def observe(*args, **kwargs):
            value = original_schedule(*args, **kwargs)
            observed.append((len(scheduler.running), blocks - scheduler.block_manager.get_num_free_gpu_blocks()))
            return value
        scheduler.schedule = observe
        llm.generate(['Explain batching.'] * 8, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True), use_tqdm=False)
        scheduler.schedule = original_schedule
        assert any(running == 8 and used > 0 for running, used in observed)
        print('PASS: eight simultaneous prompts and observed allocated blocks in the live scheduler')
    print('MEASUREMENT SMOKE TEST PASSED' + (' ON GPU' if gpu else ' (CPU/mocked interfaces only)'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--gpu', action='store_true')
    main(p.parse_args().gpu)
