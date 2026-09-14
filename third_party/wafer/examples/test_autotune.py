"""Benchmark a simple kernel so timing is tested independently of GEMM lowering."""
import math

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


def test_autotune_vector(device):
    measured = []

    def benchmark(fn, quantiles):
        times = do_bench(fn, warmup=1, rep=3, quantiles=quantiles)
        assert all(math.isfinite(t) and t > 0 for t in times)
        measured.append(times)
        return times

    @triton.autotune(configs=[triton.Config({'BLOCK': 64}), triton.Config({'BLOCK': 128})],
                     key=['N'], do_bench=benchmark)
    @triton.jit
    def add_kernel(X, Y, Out, N: tl.constexpr, BLOCK: tl.constexpr):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(X + offset, offset < N, other=0) + tl.load(Y + offset, offset < N, other=0)
        tl.store(Out + offset, value, offset < N)

    x = torch.arange(257, dtype=torch.float32, device=device)
    y = torch.full_like(x, 1.25)
    out = torch.empty_like(x)
    add_kernel[lambda meta: (triton.cdiv(x.numel(), meta['BLOCK']),)](x, y, out, N=x.numel())
    assert len(measured) == 2
    assert add_kernel.best_config.kwargs['BLOCK'] in (64, 128)
    torch.testing.assert_close(out, x + y)
