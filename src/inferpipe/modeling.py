# -*- coding: utf-8 -*-

from __future__ import annotations

import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel  # type: ignore
except Exception:  # pragma: no cover
    PeftModel = None  # type: ignore


def _resolve_model_id(model_id: str) -> tuple[str, bool]:
    """Resolve model_id to local path if present, otherwise keep HF repo id."""
    expanded = os.path.expandvars(os.path.expanduser(model_id))
    if os.path.isdir(expanded):
        return expanded, True

    looks_like_local = model_id.startswith(("/", "./", "../", "~")) or model_id.count("/") > 1
    if looks_like_local:
        raise FileNotFoundError(
            "model_id looks like a local path but was not found: "
            f"{model_id} (expanded: {expanded}). "
            "If you intended a Hugging Face repo, pass it as 'namespace/repo_name'."
        )

    return model_id, False


def _max_memory_with_headroom(headroom_gb: float) -> dict:
    """device_map=auto budget: every visible GPU minus a fixed headroom, CPU untouched."""
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory
        budget = max(int(total - headroom_gb * (1024**3)), 0)
        max_memory[i] = budget
    return max_memory


def _gpt_oss_sparse_experts_forward(self, hidden_states, router_indices=None, routing_weights=None):
    """GptOssExperts.forward restricted to the routed experts.

    transformers' GPU path computes every expert for every token (bmm over all experts), which
    needs tens of GB of transient memory per layer at prefill for 13k-token prompts on the 120B
    model (128 experts). This is the library's own CPU/training branch (identical math, only the
    top-k experts per token), applied on GPU as well. Batch size 1 inference only.
    """
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)
    num_experts = routing_weights.shape[1]
    next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts + 1)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit[:]:
        expert_idx = expert_idx[0]
        if expert_idx == num_experts:
            continue
        with torch.no_grad():
            _, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu
        out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
        weighted_output = out * routing_weights[token_idx, expert_idx, None]
        next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
    return next_states.view(batch_size, -1, self.hidden_size)


def _gpt_oss_chunked_eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """transformers' gpt-oss eager_attention_forward, computed in query chunks.

    Row-wise softmax is independent per query position, so this is numerically identical to the
    reference eager implementation; only the peak memory changes (the reference materializes the
    full heads x q_len x kv_len score matrix, >40 GB at 13k tokens). Needed because gpt-oss has no
    sdpa kernel and flex_attention's compiled block mask fails across a multi-GPU device_map.
    attn_weights are not returned (None).
    """
    from transformers.models.gpt_oss import modeling_gpt_oss as m

    chunk = int(os.getenv("ATTN_CHUNK", "1024"))
    key_states = m.repeat_kv(key, module.num_key_value_groups)
    value_states = m.repeat_kv(value, module.num_key_value_groups)
    kv_len = key_states.shape[-2]
    q_len = query.shape[-2]
    sinks = module.sinks.reshape(1, -1, 1, 1)
    outs = []
    for i in range(0, q_len, chunk):
        q = query[:, :, i : i + chunk, :]
        attn_weights = torch.matmul(q, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, i : i + chunk, :kv_len]
        combined_logits = torch.cat([attn_weights, sinks.expand(q.shape[0], -1, q.shape[-2], -1)], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probs = torch.nn.functional.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
        scores = probs[..., :-1]
        outs.append(torch.matmul(scores, value_states))
        del attn_weights, combined_logits, probs, scores
    attn_output = torch.cat(outs, dim=2).transpose(1, 2).contiguous()
    return attn_output, None


def _patch_gpt_oss_sparse_experts() -> None:
    from transformers.models.gpt_oss import modeling_gpt_oss as m

    if getattr(m.GptOssExperts.forward, "_sparse_patch", False):
        return
    _gpt_oss_sparse_experts_forward._sparse_patch = True  # type: ignore[attr-defined]
    m.GptOssExperts.forward = _gpt_oss_sparse_experts_forward
    m.eager_attention_forward = _gpt_oss_chunked_eager_attention
    print(
        "[INFO] gpt_oss: experts forward patched to routed-experts-only path; "
        f"eager attention patched to query-chunked version (ATTN_CHUNK={os.getenv('ATTN_CHUNK', '1024')})"
    )


def load_model_and_tokenizer(
    model_id: str,
    use_adapter: bool,
    adapter_path: str,
    device: str,
    quant: str = "nf4",
    gpu_headroom_gb: float = 10.0,
    attn_implementation: str = "",
):
    resolved_model_id, is_local_model = _resolve_model_id(model_id)
    tokenizer = load_tokenizer(model_id)

    device_map = "auto" if device == "auto" else None

    model_kwargs = {}
    if is_local_model:
        model_kwargs["local_files_only"] = True
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    cfg = AutoConfig.from_pretrained(resolved_model_id, **{k: v for k, v in model_kwargs.items() if k == "local_files_only"})
    if getattr(cfg, "model_type", "") == "gpt_oss":
        # Must happen BEFORE from_pretrained: with a multi-GPU device_map, accelerate captures each
        # module's forward as an instance attribute at dispatch time, so a later class patch is ignored.
        _patch_gpt_oss_sparse_experts()

    if quant == "nf4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quant == "none":
        # bf16 weights. Checkpoints shipped with MXFP4 experts (gpt-oss) are dequantized to bf16
        # explicitly, so the result does not depend on whether the `kernels` package is installed.
        qc = getattr(cfg, "quantization_config", None)
        qm = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
        if qm == "mxfp4":
            from transformers import Mxfp4Config

            model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
            print("[INFO] MXFP4 checkpoint: dequantizing experts to bf16 at load time")
        elif qm is not None:
            raise ValueError(f"--quant none does not support checkpoints quantized with {qm}")
        if device_map == "auto" and torch.cuda.is_available():
            model_kwargs["max_memory"] = _max_memory_with_headroom(gpu_headroom_gb)
    else:
        raise ValueError(f"unknown quant: {quant}")

    base = AutoModelForCausalLM.from_pretrained(
        resolved_model_id,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        use_cache=True,
        **model_kwargs,
    )
    print(
        f"[INFO] loaded {resolved_model_id} quant={quant} dtype={base.dtype} "
        f"attn={getattr(base.config, '_attn_implementation', None)} "
        f"device_map={getattr(base, 'hf_device_map', device_map)}"
    )

    model = base
    if use_adapter:
        if not adapter_path:
            raise ValueError("--use_adapter was set but --adapter_path is empty")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"adapter_path not found: {adapter_path}")
        if PeftModel is None:
            raise RuntimeError("peft is not available but --use_adapter was requested")
        model = PeftModel.from_pretrained(base, adapter_path)

    model.eval()
    return model, tokenizer


def load_tokenizer(model_id: str):
    resolved_model_id, is_local_model = _resolve_model_id(model_id)
    tokenizer_kwargs = {}
    if is_local_model:
        tokenizer_kwargs["local_files_only"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_id,
        use_fast=True,
        trust_remote_code=False,
        fix_mistral_regex=True,
        **tokenizer_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer
