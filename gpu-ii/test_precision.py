"""CPU regression tests for logical updates, partial groups and overflow replay.

The loss scaler is a controlled test double, not a GPU accuracy measurement.
The notebook additionally exercises real CUDA autocast/GradScaler on the T4.
"""
import copy
import argparse
import contextlib
from types import SimpleNamespace
from unittest.mock import patch
import torch
import hw45_training as training


class ControlledScaler:
    def __init__(self, enabled=True):
        self.enabled, self.value = enabled, 16.0
    def is_enabled(self): return self.enabled
    def scale(self, loss): return loss
    def unscale_(self, optimizer): pass
    def get_scale(self): return self.value
    def update(self, new_scale=None):
        if new_scale is not None: self.value = new_scale
    def step(self, optimizer): optimizer.step()


class TinyModel(torch.nn.Module):
    def __init__(self, fail_calls=0, dropout=0.2):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)
        self.dropout = torch.nn.Dropout(dropout)
        self.fail_calls = fail_calls
    def forward(self, input_ids, attention_mask, labels):
        loss = (self.layer(self.dropout(input_ids)) - labels).square().mean()
        if self.fail_calls:
            self.fail_calls -= 1
            loss = loss * float('nan')
        return SimpleNamespace(loss=loss)


def main(gpu=False):
    torch.manual_seed(8834)
    x, y = torch.randn(5, 2), torch.randn(5, 1)
    batches = [(x[:2], torch.ones_like(x[:2]), y[:2]),
               (x[2:3], torch.ones_like(x[2:3]), y[2:3])]
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda self, *a, **k: self))
        stack.enter_context(patch.object(torch.cuda, 'get_rng_state_all', return_value=[]))
        stack.enter_context(patch.object(torch.cuda, 'set_rng_state_all'))
        good = TinyModel()
        recovered = copy.deepcopy(good); recovered.fail_calls = 1
        original_rng = torch.get_rng_state()
        a = training.train_update(good, torch.optim.SGD(good.parameters(), lr=.01),
                                  ControlledScaler(), batches, torch.float16, False)
        torch.set_rng_state(original_rng)
        scaler = ControlledScaler()
        b = training.train_update(recovered, torch.optim.SGD(recovered.parameters(), lr=.01),
                                  scaler, batches, torch.float16, False)
        assert a[1:] == (3, 0) and b[1:] == (3, 1) and scaler.get_scale() == 8
        assert all(torch.equal(p, q) for p, q in zip(good.parameters(), recovered.parameters()))
        print('PASS: overflow replay restores dropout RNG and applies exactly the same one successful update')

        micro = TinyModel(dropout=0)
        direct = copy.deepcopy(micro)
        opt_m = torch.optim.SGD(micro.parameters(), lr=.01)
        opt_d = torch.optim.SGD(direct.parameters(), lr=.01)
        for start in (0, 3):
            group = [(x[i:i+1], torch.ones_like(x[i:i+1]), y[i:i+1]) for i in range(start, min(start+3, 5))]
            whole = [(x[start:start+3], torch.ones_like(x[start:start+3]), y[start:start+3])]
            training.train_update(micro, opt_m, ControlledScaler(False), group, torch.float16, False)
            training.train_update(direct, opt_d, ControlledScaler(False), whole, torch.float16, False)
        assert all(torch.allclose(p,q,atol=1e-7) for p,q in zip(micro.parameters(),direct.parameters()))
        print('PASS: accumulation matches direct batches, including the short final update group')

        broken = TinyModel(fail_calls=100)
        before = copy.deepcopy(broken.state_dict())
        try:
            training.train_update(broken, torch.optim.SGD(broken.parameters(), lr=.01),
                                  ControlledScaler(), batches, torch.float16, False)
        except RuntimeError as error:
            assert 'eight bounded retries' in str(error)
        else: raise AssertionError('Persistent non-finite values must not be hidden as success')
        assert all(torch.equal(before[k],v) for k,v in broken.state_dict().items())
        print('PASS: persistent overflow stops after bounded retries without updating model weights')
    print('PRECISION REGRESSION PASSED (CPU and controlled scaler only)')
    if gpu:
        assert torch.cuda.is_available(), 'GPU precision smoke requires CUDA.'
        torch.manual_seed(8834)
        reference = TinyModel().cuda()
        replay = copy.deepcopy(reference); replay.fail_calls = 1
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        ref_scaler = torch.amp.GradScaler('cuda', init_scale=16, growth_interval=100000)
        ref_result = training.train_update(reference, torch.optim.SGD(reference.parameters(), lr=.01),
                                           ref_scaler, batches, torch.float16, True)
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
        real_scaler = torch.amp.GradScaler('cuda', init_scale=16, growth_interval=100000)
        recovered = training.train_update(replay, torch.optim.SGD(replay.parameters(), lr=.01),
                                          real_scaler, batches, torch.float16, True)
        assert ref_result[2] == 0 and recovered[2] == 1 and real_scaler.get_scale() == 8
        assert all(torch.allclose(p,q,atol=2e-5,rtol=1e-4) for p,q in zip(reference.parameters(),replay.parameters()))
        print('PASS CUDA: real FP16 autocast/GradScaler recovers a deliberately non-finite attempt without losing the logical update')
        print('REAL CUDA PRECISION SMOKE PASSED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    main(parser.parse_args().gpu)
