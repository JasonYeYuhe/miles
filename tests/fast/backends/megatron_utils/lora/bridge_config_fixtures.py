from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import LlamaConfig


class _Provider:
    def __init__(self) -> None:
        self.hooks = []
        self.finalized = False
        self.ddp_config = None
        self.experimental_attention_variant = "dsv4_hybrid"

    def finalize(self) -> None:
        self.finalized = True

    def register_pre_wrap_hook(self, hook: object) -> None:
        self.hooks.append(hook)

    def provide_distributed_model(self, *, wrap_with_ddp: bool, ddp_config: object) -> list:
        assert self.finalized
        assert wrap_with_ddp
        self.ddp_config = ddp_config
        return [self]


class _DdpConfig:
    def __init__(self, *, use_distributed_optimizer: bool, grad_reduce_in_fp32: bool) -> None:
        self.use_distributed_optimizer = use_distributed_optimizer
        self.grad_reduce_in_fp32 = grad_reduce_in_fp32
        self.finalized = False

    def finalize(self) -> None:
        self.finalized = True


@pytest.fixture
def bridge_provider(monkeypatch: pytest.MonkeyPatch) -> _Provider:
    from megatron.bridge import AutoBridge
    from megatron.bridge.training import config
    from megatron.bridge.utils import fusions

    provider = _Provider()
    bridge = SimpleNamespace(
        to_megatron_provider=lambda load_weights: provider,
        _model_bridge=SimpleNamespace(),
        hf_pretrained=object(),
    )
    monkeypatch.setattr(AutoBridge, "from_hf_pretrained", lambda *args, **kwargs: bridge)
    monkeypatch.setattr(config, "DistributedDataParallelConfig", _DdpConfig)
    monkeypatch.setattr(fusions, "validate_rope_fusion_compatibility", lambda provider: False)
    return provider


@pytest.fixture
def bridge_args(tmp_path: Path) -> Namespace:
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=8,
    ).save_pretrained(tmp_path)
    return Namespace(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=3,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        virtual_pipeline_model_parallel_size=2,
        context_parallel_size=2,
        gradient_accumulation_fusion=False,
        recompute_granularity="selective",
        recompute_method="uniform",
        recompute_num_layers=2,
        recompute_modules=["core_attn"],
        distribute_saved_activations=True,
        attention_backend="flash",
        apply_rope_fusion=True,
        bias_swiglu_fusion=True,
        moe_router_dtype="fp32",
        moe_router_use_torch_mm=True,
        decoder_first_pipeline_num_layers=2,
        decoder_last_pipeline_num_layers=3,
        optimizer="adam",
        accumulate_allreduce_grads_in_fp32=True,
        dsa_kernel_backend="flash_mla",
        hf_checkpoint=tmp_path,
        enable_mtp_training=False,
        multi_lora=False,
        hf_lora_targets=["q_proj"],
        lora_type="lora",
        offload_train=False,
        dsa_attention_backend="megatron",
    )
