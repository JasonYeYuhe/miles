"""Qwen3.8-27B fully asynchronous GRPO on Terminal-Bench 2.1 via Harbor.

Requires a prepared HF checkpoint, a shared TB2.1 task directory and prompt
JSONL, Harbor with E2B support, and an eight-node Ray allocation (2 trainer,
6 rollout nodes by default). This script does not allocate machines or delete
existing jobs. See examples/experimental/harbor/README.md for preparation.

Args:
    --model_dir: Directory containing Qwen3.8-27B and its torch_dist conversion.
    --data_dir: Directory containing tasks/ and train.jsonl.
    --output_dir: Checkpoints and complete rollout/Harbor traces, grouped by run ID.
    --skip_prepare: Use an existing torch_dist conversion instead of converting.
    --dry_run: Print the proposed training arguments without starting any service.

Example:
    python examples/experimental/harbor/run_qwen38_27b.py --skip_prepare \
        --model_dir /scratch/models --data_dir /scratch/tb21 \
        --output_dir /scratch/runs --e2b_api_key_file /secrets/e2b.key
"""

import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import get_type_hints

from tap import Tap

from miles.utils.external_utils import command_utils as U
from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend
from miles.utils.external_utils.model_args_utils import _import_module_from_path
from miles.utils.workers.types import ClusterBackend

_HARBOR = _import_module_from_path(
    U.repo_base_dir / "examples/experimental/harbor/launch_common.py", "qwen38_harbor_launch_common"
)


class _ExistingRayBackend(RayCommandBackend):
    def _clean_up_previous_run(self, external_ray: bool) -> None:
        # The shared Ray backend still kills SGLang/Miles/Redis in external-Ray
        # mode. An example must not terminate unrelated processes on the host.
        if not external_ray:
            raise ValueError("This recipe requires MILES_SCRIPT_EXTERNAL_RAY=1")


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_nodes: int = 8
    num_gpus_per_node: int = 8
    trainer_nodes: int = 2
    model_dir: str = "/root/models"
    model_name: str = "Qwen3.8-27B"
    data_dir: str = "/root/datasets/tb21"
    megatron_path: str = "/root/Megatron-LM"
    skip_prepare: bool = False
    dry_run: bool = False
    num_rollout: int = 1000
    save_interval: int = 25
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 32
    max_seq_len: int = 131072
    rollout_max_response_len: int = 49152
    tensor_parallel_size: int = 4
    context_parallel_size: int = 2
    max_tokens_per_gpu: int = 8192
    learning_rate: float = 1e-6
    max_concurrent_samples: int = 128
    max_weight_staleness: int = 8
    sglang_mem_fraction_static: float = 0.8
    sglang_max_running_requests: int = 8
    session_server_workers: int = 32
    session_server_external_host: str = ""
    harbor_env_type: str = "e2b"
    harbor_env_kwargs: str = ""
    e2b_api_key_file: str = ""
    agent_model_name: str = "qwen38-27b"
    agent_timeout: int = 28800

    def __post_init__(self) -> None:
        if not 0 < self.trainer_nodes < self.num_nodes:
            raise ValueError("Reserve at least one node each for training and rollouts")
        parallel_size = self.tensor_parallel_size * self.context_parallel_size
        if parallel_size <= 0 or self.trainer_nodes * self.num_gpus_per_node % parallel_size:
            raise ValueError("Trainer GPU count must be divisible by TP * CP")
        if not 0 < self.rollout_max_response_len < self.max_seq_len:
            raise ValueError("Response budget must be positive and smaller than the sequence budget")
        if min(self.rollout_batch_size, self.n_samples_per_prompt, self.save_interval, self.num_rollout) <= 0:
            raise ValueError("Batch sizes, rollout count and checkpoint interval must be positive")

    def create_backend(self) -> RayCommandBackend:
        if self.cluster_backend != ClusterBackend.RAY:
            raise ValueError("This example supports an existing Ray allocation only")
        return _ExistingRayBackend(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_id

    @property
    def hf_checkpoint(self) -> Path:
        return Path(self.model_dir) / self.model_name

    @property
    def ref_load(self) -> Path:
        return Path(self.model_dir) / f"{self.model_name}_torch_dist"

    @property
    def harbor_tasks_dir(self) -> str:
        return str(Path(self.data_dir) / "tasks")

    @property
    def harbor_trials_dir(self) -> str:
        return str(self.run_dir / "harbor_trials")


def _build_train_args(args: ScriptArgs) -> str:
    checkpoint = (
        f"--hf-checkpoint {shlex.quote(str(args.hf_checkpoint))} "
        f"--ref-load {shlex.quote(str(args.ref_load))} "
        f"--load {shlex.quote(str(args.run_dir / 'checkpoints'))} "
        f"--save {shlex.quote(str(args.run_dir / 'checkpoints'))} --save-interval {args.save_interval} "
    )
    rollout = (
        f"--prompt-data {shlex.quote(str(Path(args.data_dir) / 'train.jsonl'))} "
        "--input-key prompt --metadata-key metadata --rollout-shuffle "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        f"--max-seq-len {args.max_seq_len} --rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 --balance-data "
        "--apply-chat-template-kwargs "
        '\'{"enable_thinking": true, "preserve_thinking": true, "reasoning_effort": "medium"}\' '
    )
    performance = (
        f"--tensor-model-parallel-size {args.tensor_parallel_size} "
        f"--context-parallel-size {args.context_parallel_size} --pipeline-model-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 --sequence-parallel "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    )
    algorithm = (
        "--advantage-estimator grpo --kl-loss-coef 0 --entropy-coef 0 "
        "--eps-clip 0.2 --eps-clip-high 0.28 --use-tis --tis-clip 2 "
        "--fully-async --async-data-buffer-capacity-factor 2 --async-unused-samples-handler drop "
        f"--async-max-concurrent-samples {args.max_concurrent_samples} "
        f"--max-weight-staleness {args.max_weight_staleness} "
        "--pause-generation-mode in_place --update-weight-transfer-mode broadcast --update-weights-interval 1 "
    )
    optimizer = (
        f"--optimizer adam --lr {args.learning_rate} --lr-decay-style constant "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )
    sglang = (
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-max-running-requests {args.sglang_max_running_requests} "
        f"--sglang-context-length {args.max_seq_len} --sglang-chunked-prefill-size 8192 "
        "--sglang-router-policy cache_aware --router-balance-abs-threshold 4 --sglang-router-port 31000 "
        f"--sglang-served-model-name {args.agent_model_name} --sglang-reasoning-parser qwen3 "
        "--sglang-tool-call-parser qwen3_coder "
    )
    # FullyAsyncRolloutFn owns the rollout loop; the shared metrics RolloutFn
    # cannot also be selected. Keep the common agent/session/reward wiring.
    agent = _HARBOR.agentic_train_args(
        tito_model="qwen38small",
        session_server_workers=args.session_server_workers,
        session_server_external_host=args.session_server_external_host,
    ).replace("--rollout-function-path generate.RolloutFn ", "")
    misc = (
        f"--actor-num-nodes {args.trainer_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {(args.num_nodes - args.trainer_nodes) * args.num_gpus_per_node} "
        "--attention-dropout 0 --hidden-dropout 0 --attention-backend flash "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
        "--use-miles-dashboard --observe-training-entropy --use-rollout-entropy --log-multi-turn "
        f"--dump-details {shlex.quote(str(args.run_dir / 'traces'))} "
    )
    return checkpoint + rollout + performance + algorithm + optimizer + sglang + agent + misc


def _runtime_env(args: ScriptArgs) -> dict[str, str]:
    # Explicit run-wide budgets override inherited shell defaults. This recipe
    # deliberately keeps append-only trajectories (no summarization).
    return {
        **_HARBOR.harbor_env_vars(args),
        "PYTHONPATH": ":".join([args.megatron_path, *_HARBOR.agentic_pythonpath_dirs(), str(U.repo_base_dir)]),
        "AGENT_MAX_INPUT_TOKENS": str(args.max_seq_len - args.rollout_max_response_len),
        "AGENT_MAX_OUTPUT_TOKENS": str(args.rollout_max_response_len),
        "HARBOR_MAX_SEQ_LEN": str(args.max_seq_len),
        "HARBOR_TERMINUS_2_ENABLE_SUMMARIZE": "false",
        "HARBOR_TERMINUS_2_LINEAR_HISTORY": "true",
        "HARBOR_RESPONSE_LENGTH_POLICY": "abort",
        "AGENT_TRIAL_TIMEOUT": str(args.agent_timeout + 1800),
    }


def prepare(args: ScriptArgs) -> None:
    args.create_backend().convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type="qwen3.8-27B",
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=str(args.hf_checkpoint),
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs) -> None:
    train_args = _build_train_args(args)
    if args.dry_run:
        print(
            json.dumps(
                {"run_id": args.run_id, "train_script": "train_async.py", "train_args": shlex.split(train_args)},
                indent=2,
            )
        )
        return
    train_args += U.get_default_wandb_args(__file__, run_id=args.run_id)
    args.create_backend().execute_train(
        train_args=train_args,
        config=args,
        train_script="train_async.py",
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3.8-27B",
        megatron_path=args.megatron_path,
        extra_env_vars=_runtime_env(args),
    )


def main(args: ScriptArgs) -> None:
    if not args.dry_run and os.environ.get("MILES_SCRIPT_EXTERNAL_RAY") != "1":
        raise ValueError(
            "Join the allocated nodes to Ray first, then set MILES_SCRIPT_EXTERNAL_RAY=1. "
            "This recipe refuses the launcher's default host-wide Ray/SGLang cleanup."
        )
    if not args.skip_prepare and not args.dry_run:
        prepare(args)
    execute(args)


def _parse_args() -> ScriptArgs:
    # Resolve inherited postponed annotations before giving them to Tap. All
    # CLI defaults still come from the dataclass, including the fresh run ID.
    defaults = ScriptArgs()
    cli_type = type("RecipeCLI", (Tap,), {"__annotations__": get_type_hints(ScriptArgs), **vars(defaults)})
    return ScriptArgs(**cli_type().parse_args().as_dict())


if __name__ == "__main__":
    main(_parse_args())
