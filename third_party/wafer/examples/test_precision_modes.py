"""Integer precision modes and reciprocal correction on the real device."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def integer_math(X, Y, A, Q, R, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(X + i)
    y = tl.load(Y + i)
    tl.store(A + i, x + y)
    tl.store(Q + i, x // y)
    tl.store(R + i, x % y)


@pytest.mark.parametrize("mode,dtype", [(0, torch.int32), (1, torch.int64), (2, torch.int32)])
def test_integer_modes(device, mode, dtype):
    # Mode 0 checks exactly representable inputs, including reciprocal rounding
    # at equal operands. Modes 1/2 additionally exercise integers beyond 2**24.
    values = [7, -7, 14, -14, 15, -15, 0, 1]
    if mode:
        values[4:6] = [2**24 + 7, -(2**24 + 7)]
    x = torch.tensor(values, dtype=dtype, device=device)
    y = torch.tensor([7, 7, -7, -7, 7, 7, 7, 7], dtype=dtype, device=device)
    outputs = [torch.empty_like(x) for _ in range(3)]
    integer_math[(1,)](x, y, *outputs, 8, precision_mode=mode)
    references = (x + y, torch.div(x, y, rounding_mode="trunc"), x - torch.div(x, y, rounding_mode="trunc") * y)
    for actual, expected in zip(outputs, references):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
