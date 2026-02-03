import torch
import pytest

from apps.backend.runtime.ops.operations import using_codex_operations
from apps.backend.runtime.common.nn.t5 import T5LayerNorm
from apps.backend.runtime.families.flux.embed import EmbedND
from apps.backend.runtime.families.flux.components import RMSNorm as FluxRMSNorm
from apps.backend.runtime.families.flux.geometry import apply_rotary_embeddings
from apps.backend.runtime.families.sd.mmditx import RMSNorm as SdRMSNorm
from apps.backend.runtime.families.wan22.model import WanRMSNorm
from apps.backend.runtime.families.wan22.model import WanSelfAttention
from apps.backend.runtime.families.zimage.model import RMSNorm as ZImageRMSNorm
from apps.backend.runtime.families.zimage.model import apply_rotary_emb
from apps.backend.runtime.families.zimage.qwen3 import RMSNorm as Qwen3RMSNorm
from apps.backend.runtime.families.zimage.qwen3 import RotaryEmbedding, apply_rotary_pos_emb


@pytest.mark.parametrize("manual_cast_enabled", [False, True])
@pytest.mark.parametrize("weight_format", [None, "gguf"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ops_layer_norm_fp32_compute_dtype_preserved(dtype: torch.dtype, weight_format: str | None, manual_cast_enabled: bool) -> None:
    device = torch.device("cpu")
    with using_codex_operations(
        device=device,
        dtype=dtype,
        manual_cast_enabled=manual_cast_enabled,
        weight_format=weight_format,
    ):
        ln = torch.nn.LayerNorm(16, elementwise_affine=True)
        x = torch.randn(2, 4, 16, device=device, dtype=dtype)
        assert ln.weight is not None
        assert ln.bias is not None
        ln.weight.data.fill_(1.0)
        ln.bias.data.zero_()
        with torch.no_grad():
            out = ln(x)
        assert out.dtype == x.dtype
        assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("manual_cast_enabled", [False, True])
@pytest.mark.parametrize("weight_format", [None, "gguf"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ops_group_norm_fp32_compute_dtype_preserved(dtype: torch.dtype, weight_format: str | None, manual_cast_enabled: bool) -> None:
    device = torch.device("cpu")
    with using_codex_operations(
        device=device,
        dtype=dtype,
        manual_cast_enabled=manual_cast_enabled,
        weight_format=weight_format,
    ):
        gn = torch.nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True)
        x = torch.randn(2, 16, 8, 8, device=device, dtype=dtype)
        assert gn.weight is not None
        assert gn.bias is not None
        gn.weight.data.fill_(1.0)
        gn.bias.data.zero_()
        with torch.no_grad():
            out = gn(x)
        assert out.dtype == x.dtype
        assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("weight_format", [None, "gguf"])
def test_ops_layer_norm_applies_online_lora(weight_format: str | None) -> None:
    device = torch.device("cpu")
    dtype = torch.float16
    with using_codex_operations(device=device, dtype=dtype, manual_cast_enabled=False, weight_format=weight_format):
        ln = torch.nn.LayerNorm(16, elementwise_affine=True)
        x = torch.randn(2, 4, 16, device=device, dtype=dtype)
        assert ln.weight is not None
        assert ln.bias is not None
        ln.weight.data.fill_(1.0)
        ln.bias.data.zero_()
        with torch.no_grad():
            base = ln(x).float()
        delta = torch.ones_like(ln.weight, dtype=torch.float32)
        ln.codex_online_loras = {"weight": [(1.0, delta, 1.0, None, None)]}
        with torch.no_grad():
            patched = ln(x).float()
        assert not torch.allclose(base, patched)


def test_ops_norms_reject_non_float_inputs() -> None:
    device = torch.device("cpu")
    with using_codex_operations(device=device, dtype=torch.float16, manual_cast_enabled=False):
        ln = torch.nn.LayerNorm(16, elementwise_affine=True)
        with pytest.raises(TypeError):
            _ = ln(torch.zeros(2, 4, 16, device=device, dtype=torch.int64))

        gn = torch.nn.GroupNorm(num_groups=4, num_channels=16, eps=1e-6, affine=True)
        with pytest.raises(TypeError):
            _ = gn(torch.zeros(2, 16, 8, 8, device=device, dtype=torch.int64))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_t5_layer_norm_fp32_compute_dtype_preserved(dtype: torch.dtype) -> None:
    device = torch.device("cpu")
    ln = T5LayerNorm(16)
    x = torch.randn(2, 4, 16, device=device, dtype=dtype)
    ln.weight.data.fill_(1.0)
    with torch.no_grad():
        out = ln(x)
    assert out.dtype == x.dtype
    assert torch.isfinite(out.float()).all()


def test_t5_layer_norm_rejects_non_float_input() -> None:
    ln = T5LayerNorm(16)
    with pytest.raises(TypeError):
        _ = ln(torch.zeros(2, 4, 16, dtype=torch.int64))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_flux_rmsnorm_dtype_preserved(dtype: torch.dtype) -> None:
    device = torch.device("cpu")
    norm = FluxRMSNorm(16)
    x = torch.randn(2, 4, 16, device=device, dtype=dtype)
    with torch.no_grad():
        out = norm(x)
    assert out.dtype == x.dtype
    assert torch.isfinite(out.float()).all()


def test_flux_rmsnorm_rejects_non_float_input() -> None:
    norm = FluxRMSNorm(16)
    with pytest.raises(TypeError):
        _ = norm(torch.zeros(2, 4, 16, dtype=torch.int64))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_wan_rmsnorm_dtype_preserved(dtype: torch.dtype) -> None:
    device = torch.device("cpu")
    norm = WanRMSNorm(16)
    x = torch.randn(2, 4, 16, device=device, dtype=dtype)
    with torch.no_grad():
        out = norm(x)
    assert out.dtype == x.dtype
    assert torch.isfinite(out.float()).all()


def test_wan_rmsnorm_rejects_non_float_input() -> None:
    norm = WanRMSNorm(16)
    with pytest.raises(TypeError):
        _ = norm(torch.zeros(2, 4, 16, dtype=torch.int64))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sd_rmsnorm_dtype_preserved(dtype: torch.dtype) -> None:
    device = torch.device("cpu")
    norm = SdRMSNorm(16, elementwise_affine=True)
    assert norm.weight is not None
    norm.weight.data.fill_(1.0)
    x = torch.randn(2, 4, 16, device=device, dtype=dtype)
    with torch.no_grad():
        out = norm(x)
    assert out.dtype == x.dtype
    assert torch.isfinite(out.float()).all()


def test_sd_rmsnorm_rejects_non_float_input() -> None:
    norm = SdRMSNorm(16, elementwise_affine=False)
    with pytest.raises(TypeError):
        _ = norm(torch.zeros(2, 4, 16, dtype=torch.int64))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_qwen3_rmsnorm_dtype_preserved(dtype: torch.dtype) -> None:
    device = torch.device("cpu")
    norm = Qwen3RMSNorm(16)
    x = torch.randn(2, 4, 16, device=device, dtype=dtype)
    with torch.no_grad():
        out = norm(x)
    assert out.dtype == x.dtype
    assert torch.isfinite(out.float()).all()


def test_qwen3_rmsnorm_rejects_non_float_input() -> None:
    norm = Qwen3RMSNorm(16)
    with pytest.raises(TypeError):
        _ = norm(torch.zeros(2, 4, 16, dtype=torch.int64))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_zimage_rmsnorm_dtype_preserved(dtype: torch.dtype) -> None:
    device = torch.device("cpu")
    norm = ZImageRMSNorm(16)
    x = torch.randn(2, 4, 16, device=device, dtype=dtype)
    with torch.no_grad():
        out = norm(x)
    assert out.dtype == x.dtype
    assert torch.isfinite(out.float()).all()


def test_zimage_rmsnorm_rejects_non_float_input() -> None:
    norm = ZImageRMSNorm(16)
    with pytest.raises(TypeError):
        _ = norm(torch.zeros(2, 4, 16, dtype=torch.int64))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_flux_rope_dtype_preserved(dtype: torch.dtype) -> None:
    device = torch.device("cpu")
    batch = 2
    heads = 4
    seq_len = 8
    head_dim = 16
    ids = torch.arange(seq_len, device=device, dtype=torch.long).view(1, seq_len, 1).repeat(batch, 1, 1)
    pe = EmbedND(dim=head_dim, theta=10000, axes_dim=(head_dim,))(ids)
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    q_out, k_out = apply_rotary_embeddings(q, k, pe)
    assert q_out.dtype == q.dtype
    assert k_out.dtype == k.dtype
    assert torch.isfinite(q_out.float()).all()
    assert torch.isfinite(k_out.float()).all()


def test_flux_rope_rejects_non_float_qk() -> None:
    batch = 1
    heads = 1
    seq_len = 4
    head_dim = 16
    ids = torch.arange(seq_len, dtype=torch.long).view(1, seq_len, 1)
    pe = EmbedND(dim=head_dim, theta=10000, axes_dim=(head_dim,))(ids)
    q = torch.zeros(batch, heads, seq_len, head_dim, dtype=torch.int64)
    k = torch.zeros(batch, heads, seq_len, head_dim, dtype=torch.int64)
    with pytest.raises(TypeError):
        _ = apply_rotary_embeddings(q, k, pe)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_qwen3_rope_dtype_preserved(dtype: torch.dtype) -> None:
    device = torch.device("cpu")
    batch = 2
    heads = 4
    seq_len = 8
    head_dim = 16
    rot = RotaryEmbedding(head_dim, max_position_embeddings=128, base=10000.0)
    dummy = torch.zeros(batch, seq_len, head_dim, device=device, dtype=dtype)
    cos, sin = rot(dummy, seq_len)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_out.dtype == q.dtype
    assert k_out.dtype == k.dtype
    assert torch.isfinite(q_out.float()).all()
    assert torch.isfinite(k_out.float()).all()


def test_qwen3_rope_rejects_non_float_inputs() -> None:
    device = torch.device("cpu")
    batch = 1
    heads = 1
    seq_len = 4
    head_dim = 16
    rot = RotaryEmbedding(head_dim, max_position_embeddings=128, base=10000.0)
    dummy = torch.zeros(batch, seq_len, head_dim, device=device, dtype=torch.float16)
    cos, sin = rot(dummy, seq_len)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = torch.zeros(batch, heads, seq_len, head_dim, device=device, dtype=torch.int64)
    k = torch.zeros(batch, heads, seq_len, head_dim, device=device, dtype=torch.int64)
    with pytest.raises(TypeError):
        _ = apply_rotary_pos_emb(q, k, cos, sin)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_wan_rope_dtype_preserved(dtype: torch.dtype) -> None:
    device = torch.device("cpu")
    batch = 2
    seq_len = 8
    heads = 4
    head_dim = 16
    attn = WanSelfAttention(dim=heads * head_dim, num_heads=heads, qkv_bias=True)
    hidden = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=dtype)
    freqs_cos = torch.randn(1, seq_len, 1, head_dim, device=device, dtype=torch.float32)
    freqs_sin = torch.randn(1, seq_len, 1, head_dim, device=device, dtype=torch.float32)
    out = attn._apply_rope(hidden, freqs_cos, freqs_sin)
    assert out.dtype == hidden.dtype
    assert torch.isfinite(out.float()).all()


def test_wan_rope_rejects_non_float_hidden_states() -> None:
    device = torch.device("cpu")
    heads = 4
    head_dim = 16
    attn = WanSelfAttention(dim=heads * head_dim, num_heads=heads, qkv_bias=True)
    hidden = torch.zeros(2, 8, heads, head_dim, device=device, dtype=torch.int64)
    freqs_cos = torch.zeros(1, 8, 1, head_dim, device=device, dtype=torch.float32)
    freqs_sin = torch.zeros(1, 8, 1, head_dim, device=device, dtype=torch.float32)
    with pytest.raises(TypeError):
        _ = attn._apply_rope(hidden, freqs_cos, freqs_sin)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_zimage_rope_dtype_preserved(dtype: torch.dtype) -> None:
    device = torch.device("cpu")
    batch = 2
    seq_len = 8
    heads = 4
    head_dim = 16
    x = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=dtype)
    freqs = torch.ones(batch, seq_len, head_dim // 2, device=device, dtype=torch.complex64)
    out = apply_rotary_emb(x, freqs)
    assert out.dtype == x.dtype
    assert torch.isfinite(out.float()).all()


def test_zimage_rope_rejects_non_float_input() -> None:
    batch = 1
    seq_len = 4
    heads = 1
    head_dim = 16
    x = torch.zeros(batch, seq_len, heads, head_dim, dtype=torch.int64)
    freqs = torch.ones(batch, seq_len, head_dim // 2, dtype=torch.complex64)
    with pytest.raises(TypeError):
        _ = apply_rotary_emb(x, freqs)
