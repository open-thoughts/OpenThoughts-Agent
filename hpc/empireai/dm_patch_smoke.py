"""Unit smoke for the transformers-5.x DenseMixer Qwen3-MoE port.

Builds a TINY real `Qwen3MoeSparseMoeBlock`, captures the stock (sparse) forward,
applies `apply_qwen3_moe_patch()`, and checks:
  1. patched forward runs with no AttributeError (the 5.x API break is fixed);
  2. output shape is preserved and returned as a bare TENSOR (not a tuple);
  3. STE property: patched forward VALUE == stock sparse forward value;
  4. backward produces FINITE, non-trivial grads on the router (gate.weight) AND
     the experts (gate_up_proj) — i.e. the dense router gradient flows.
Runs in seconds on 1 GPU. Exits non-zero on any failure so the build gate fails loud.
"""

import sys
import torch

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32  # fp32 for a clean numeric compare

cfg = Qwen3MoeConfig(
    hidden_size=64,
    moe_intermediate_size=32,
    num_experts=8,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    hidden_act="silu",
)
block = Qwen3MoeSparseMoeBlock(cfg).to(dev).to(dtype)
for p in block.parameters():
    torch.nn.init.normal_(p, std=0.02)

x = torch.randn(4, 16, 64, device=dev, dtype=dtype)

# --- stock (sparse) forward reference, BEFORE patching ---
with torch.no_grad():
    ref = block(x.clone())
assert isinstance(ref, torch.Tensor), f"stock forward returned {type(ref)}, expected Tensor"

# --- apply the DenseMixer patch (class-level forward swap) ---
from densemixer.patching import apply_qwen3_moe_patch  # noqa: E402
apply_qwen3_moe_patch()

xin = x.clone().requires_grad_(True)
out = block(xin)
assert isinstance(out, torch.Tensor), f"patched forward returned {type(out)}, expected Tensor (5.x API)"
assert out.shape == ref.shape, f"shape {tuple(out.shape)} != {tuple(ref.shape)}"

# STE: forward VALUE must equal the stock sparse output
max_diff = (out - ref).abs().max().item()
assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3), f"STE forward-value mismatch, max_diff={max_diff:.3e}"

# backward: finite, non-trivial grads on router + experts
loss = out.float().pow(2).mean()
loss.backward()
gw = block.gate.weight.grad
eg = block.experts.gate_up_proj.grad
assert gw is not None and torch.isfinite(gw).all(), "gate.weight.grad missing/non-finite"
assert eg is not None and torch.isfinite(eg).all(), "experts.gate_up_proj.grad missing/non-finite"
rows_with_grad = (gw.abs().sum(dim=1) > 0).sum().item()
assert gw.abs().sum().item() > 0, "router grad is all-zero"
assert eg.abs().sum().item() > 0, "expert grad is all-zero"

print(f"DM_SMOKE_OK max_diff={max_diff:.3e} router_rows_with_grad={rows_with_grad}/{cfg.num_experts} "
      f"grad_gate_norm={gw.norm().item():.4f} grad_expert_norm={eg.norm().item():.4f}")
sys.exit(0)
