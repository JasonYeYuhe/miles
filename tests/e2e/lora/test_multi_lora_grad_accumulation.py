"""Two identical requests accumulate exactly twice one request's expert-adapter gradient under EP."""

import argparse
import os
import subprocess
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.bridge.peft.multi_lora_layers import ExpertSlotRouting, MultiLoRAGroupedExpertLinear, init_adapter_slot
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear, TERowParallelGroupedLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci

from miles.backends.megatron_utils.lora.optimizer import accumulate_gradients, adapter_slot_parameters

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["lora", "multi-lora"], hardware=["hopper"])

N_ADAPTERS = 2


class ExpertModel(MegatronModule):
    def __init__(self, config):
        super().__init__(config)
        self.layers = torch.nn.ModuleList()
        for index, (linear, inputs, outputs) in enumerate(
            ((TEColumnParallelGroupedLinear, 32, 64), (TERowParallelGroupedLinear, 32, 32))
        ):
            base = linear(
                num_gemms=2,
                input_size=inputs,
                output_size=outputs,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=True,
                is_expert=True,
            )
            base.requires_grad_(False)
            self.layers.append(
                MultiLoRAGroupedExpertLinear(
                    base,
                    n_adapters=N_ADAPTERS,
                    dim=8,
                    alpha=16,
                    full_name=f"layers.{index}.linear_fc{index + 1}",
                    num_local_experts=2,
                    row_init_method="normal",
                )
            )

    def forward(self, inputs):
        # Tokens arrive in expert order with the two slots interleaved.
        slots = torch.arange(inputs.shape[0], device=inputs.device) % N_ADAPTERS
        experts = torch.arange(inputs.shape[0], device=inputs.device) // (inputs.shape[0] // 2)
        keys = slots * 2 + experts
        order = keys.argsort(stable=True)
        counts = torch.bincount(keys, minlength=N_ADAPTERS * 2)
        routing = ExpertSlotRouting(
            order,
            order.argsort(),
            counts.cumsum(0, dtype=torch.int32),
            counts.view(N_ADAPTERS, 2).sum(1),
            inputs.shape[0],
        )
        for index, layer in enumerate(self.layers):
            layer.expert_slot_routing = routing
            inputs, _ = layer(inputs, [inputs.shape[0] // 2] * 2)
            if index == 0:
                gate, up = inputs.chunk(2, dim=-1)
                inputs = F.silu(gate) * up
        return inputs


def forward_backward(model, inputs):
    with accumulate_gradients(model):
        model[0](inputs).float().square().sum().backward()
        finalize_model_grads(model)


def run_worker():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    parallel_state.initialize_model_parallel(expert_model_parallel_size=2, expert_tensor_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=False,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        params_dtype=torch.bfloat16,
        bf16=True,
        gradient_accumulation_fusion=False,
    )
    model = [
        DistributedDataParallel(
            config,
            DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False),
            ExpertModel(config).cuda(),
        )
    ]
    for slot in range(N_ADAPTERS):
        init_adapter_slot(model, slot, rank=8, alpha=16, seed=100 + slot)
    params = [p for slot in range(N_ADAPTERS) for p in adapter_slot_parameters(model, slot)]

    inputs = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
    forward_backward(model, inputs)
    once = [p.main_grad.clone() for p in params]
    assert all(torch.count_nonzero(g) > 0 for g in once)
    forward_backward(model, inputs)
    for param, expected in zip(params, once, strict=True):
        torch.testing.assert_close(param.main_grad, expected * 2, atol=0, rtol=0)

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    if parser.parse_args().worker:
        run_worker()
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                __file__,
                "--worker",
            ],
            check=True,
        )
