"""Exercise the hardware E5M2 -> FP16 CRT path for all raw encodings."""
import torch
import triton
import triton.language as tl


@triton.jit
def convert_e5m2_kernel(src, dst):
    indices = tl.arange(0, 256)
    tl.store(dst + indices, tl.load(src + indices).to(tl.float16))


def test_e5m2_to_fp16_all_encodings(device):
    raw = torch.arange(256, dtype=torch.uint8, device=device)
    out = torch.empty(256, dtype=torch.float16, device=device)
    convert_e5m2_kernel[(1,)](triton.reinterpret(raw, tl.float8e5), out)
    expected = raw.cpu().view(torch.float8_e5m2).to(torch.float16)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    # Torch may quiet NaNs. Test the CRT's payload/sign preservation separately.
    bits = out.cpu().view(torch.uint16).to(torch.int32)
    codes = raw.cpu().to(torch.int32)
    assert torch.equal(bits >> 15, codes >> 7)
    nan = ((codes & 0x7c) == 0x7c) & ((codes & 3) != 0)
    assert torch.equal(bits[nan] & 1023, (codes[nan] & 3) * 256)
