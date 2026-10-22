import shlex
import sys
from dataclasses import dataclass, field

import pytest
import torch
from tests.fast.launch_scripts.model_args_harness import expand_model_args
from tests.fast.launch_scripts.py_harness import import_launch_script
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

_LAUNCHER = "scripts/run_muse_spark_open.py"
_PARTIAL = "Muse-Spark-Open-last4-partial"
# Hardware -> (num_nodes, rollout GPUs per engine, max tokens per GPU, mem fraction, running requests, graph bs).
_HARDWARE = {
    "H200": (8, 16, 4096, 0.8, 64, [1, 2, 4, 8, 16, 32, 64]),
    "B300": (4, 8, 8192, 0.83, 128, [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]),
}
_LAYOUT_FIELDS = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "decoder_first_last_pipeline_num_layers",
)


@dataclass
class _Backend:
    commands: list = field(default_factory=list)
    conversions: list = field(default_factory=list)
    datasets: list = field(default_factory=list)
    trains: list = field(default_factory=list)

    def exec_command_cpu(self, command):
        self.commands.append(command)

    def convert_checkpoint(self, **kwargs):
        self.conversions.append(kwargs)

    def hf_download_dataset(self, full_name, data_dir):
        self.datasets.append(full_name)

    def execute_train(self, **kwargs):
        self.trains.append(kwargs)


@pytest.fixture
def launcher():
    return import_launch_script(REPO_ROOT / _LAUNCHER)


def _run(monkeypatch, launcher, **kwargs) -> _Backend:
    backend = _Backend()
    monkeypatch.setattr(launcher.ScriptArgs, "create_backend", lambda self: backend)
    launcher.main(**kwargs)
    return backend


def _train_args(monkeypatch, launcher, **kwargs) -> tuple[list[str], dict]:
    (call,) = _run(monkeypatch, launcher, **kwargs).trains
    return shlex.split(call["train_args"]), call


def _validate(monkeypatch, tokens, *, megatron_model_type, world_size, blackwell):
    from miles.backends.megatron_utils.arguments import parse_args, set_default_megatron_args, validate_args
    from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3) if blackwell else (9, 0))
    monkeypatch.setattr("megatron.training.arguments.get_device_arch_version", lambda: 10 if blackwell else 9)
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    monkeypatch.setattr(sys, "argv", ["train.py", *expand_model_args(megatron_model_type), *tokens])
    args = parse_args(extra_args_provider=get_miles_extra_args_provider())
    args.rank, args.world_size, args.compress_ratios = 0, world_size, None
    args = set_default_megatron_args(args)
    miles_validate_args(args)
    validate_args(args)
    return args


# Layouts as (TP, PP, CP, EP, layers on each edge stage); None keeps the hardware recipe's.
@pytest.mark.parametrize(
    "hardware,layout,optimizer",
    [
        ("H200", None, "adam"),
        ("H200", (4, 8, 2, 8, 7), "muon"),
        ("H200", (4, 4, 4, 16, 15), "muon"),
        ("B300", None, "adam"),
        ("B300", None, "muon"),
    ],
)
@pytest.mark.parametrize("counts", [(3, 8, 8), (100, 32, 16)])
def test_dapo_recipe_survives_megatron_validation(
    monkeypatch, tmp_path, launcher, counts, hardware, layout, optimizer
):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    template = tmp_path / "chat.jinja"
    template.write_text("{{ messages }}")
    if layout is None:
        defaults = launcher.ScriptArgs(hardware=hardware)
        layout = tuple(getattr(defaults, name) for name in _LAYOUT_FIELDS)
    use_r3 = hardware == "B300" or layout[2] > 1
    tokens, call = _train_args(
        monkeypatch,
        launcher,
        hardware=hardware,
        **dict(zip(_LAYOUT_FIELDS, layout, strict=True)),
        optimizer=optimizer,
        reasoning_effort="xhigh" if optimizer == "muon" else None,
        rollout_max_response_len=4096 if optimizer == "muon" else 2048,
        use_rollout_routing_replay=use_r3,
        num_rollout=counts[0],
        rollout_batch_size=counts[1],
        n_samples_per_prompt=counts[2],
        chat_template_path=str(template),
    )
    num_nodes, gpus_per_engine, max_tokens, mem_fraction, running, graph_bs = _HARDWARE[hardware]
    assert "--use-dynamic-batch-size" in tokens and "--micro-batch-size" not in tokens
    assert call["num_gpus_per_node"] == 8 and call["config"].num_nodes == num_nodes
    args = _validate(
        monkeypatch,
        tokens,
        megatron_model_type="muse-spark-open",
        world_size=8 * num_nodes,
        blackwell=hardware == "B300",
    )
    assert args.hf_checkpoint == "/root/models/Muse-Spark-Open-816B-A42B"
    assert args.ref_load == "/root/models/Muse-Spark-Open-816B-A42B_torch_dist"
    assert args.prompt_data == "/root/datasets/dapo-math-17k/dapo-math-17k.jsonl"
    assert (args.rollout_batch_size, args.n_samples_per_prompt, args.global_batch_size) == (
        counts[1],
        counts[2],
        counts[1] * counts[2],
    )
    assert args.rollout_max_response_len == (4096 if optimizer == "muon" else 2048)
    if optimizer == "muon":
        assert args.apply_chat_template_kwargs["reasoning_effort"] == "xhigh"
    assert args.use_dynamic_batch_size and args.max_tokens_per_gpu == max_tokens
    assert args.use_rollout_routing_replay == use_r3
    assert args.use_routing_replay == args.use_rollout_routing_replay
    assert args.num_rollout == counts[0] and args.num_steps_per_rollout == 1
    assert args.dynamic_sampling_filter_path is None and not args.use_wandb
    tp, pp, cp, ep, edge_layers = layout
    assert (args.tensor_model_parallel_size, args.pipeline_model_parallel_size, args.expert_model_parallel_size) == (
        tp,
        pp,
        ep,
    )
    assert args.context_parallel_size == cp
    if cp > 1:
        from megatron.training.arguments import core_transformer_config_from_args

        from miles_plugins.models.muse_spark.provider import _validate_fused_context_parallel

        config = core_transformer_config_from_args(args)
        assert config.cp_comm_type == "a2a"
        _validate_fused_context_parallel(config)
    assert args.data_parallel_size == 1 and args.expert_tensor_parallel_size == 1
    assert args.num_layers == 62 and args.moe_layer_freq == [1] * 62
    assert args.decoder_first_pipeline_num_layers == args.decoder_last_pipeline_num_layers == edge_layers
    if optimizer == "adam":
        assert args.optimizer == "adam" and args.stream_optimizer_state_to_disk
        assert args.stream_optimizer_state_moment_dtype == "fp32" and args.offload_train_target == "disk"
    else:
        assert args.optimizer == "muon" and args.use_layer_wise_distributed_optimizer
        assert not args.stream_optimizer_state_to_disk
        assert args.offload_train_target == "cpu" and args.chunked_optimizer_state_offload
        assert args.chunked_muon_master_weights and args.optimizer_state_offload_fraction == 1
    assert args.loss_type == "policy_loss" and args.advantage_estimator == "grpo"
    assert args.calculate_per_token_loss and args.skip_actor_forward_only
    assert args.save is None and not args.debug_train_only and not args.debug_rollout_only
    assert (args.actor_num_nodes, args.actor_num_gpus_per_node) == (num_nodes, 8)
    assert (args.rollout_num_gpus_per_engine, args.sglang_ep_size, args.sglang_pp_size) == (
        gpus_per_engine,
        gpus_per_engine,
        1,
    )
    assert not args.sglang_disable_cuda_graph and args.sglang_mem_fraction_static == mem_fraction
    assert args.sglang_max_running_requests == running and args.sglang_cuda_graph_bs_decode == graph_bs
    assert args.save_debug_rollout_data.endswith("/rollout_data/{rollout_id}.pt")
    assert args.pin_rollout_manager_to_head == (hardware == "B300")


@pytest.mark.parametrize("hardware,layout", [("H200", (8, 8, 1, 8, 7)), ("B300", (8, 4, 1, 8, 15))])
def test_hardware_sets_the_cluster_and_engine_sizing(launcher, hardware, layout):
    args = launcher.ScriptArgs(hardware=hardware)
    num_nodes, gpus_per_engine, max_tokens, mem_fraction, running, graph_bs = _HARDWARE[hardware]
    assert (args.num_nodes, args.num_gpus_per_node, args.rollout_num_gpus_per_engine) == (
        num_nodes,
        8,
        gpus_per_engine,
    )
    assert tuple(getattr(args, name) for name in _LAYOUT_FIELDS) == layout
    assert (args.max_tokens_per_gpu, args.sglang_mem_fraction_static, args.sglang_max_running_requests) == (
        max_tokens,
        mem_fraction,
        running,
    )
    assert list(args.sglang_cuda_graph_bs_decode) == graph_bs


def test_explicit_values_override_the_hardware_recipe(launcher):
    args = launcher.ScriptArgs(
        hardware="B300",
        max_tokens_per_gpu=4096,
        sglang_mem_fraction_static=0.8,
        sglang_max_running_requests=64,
        sglang_cuda_graph_bs_decode=(1, 2),
        pin_rollout_manager_to_head=False,
    )
    assert (args.max_tokens_per_gpu, args.sglang_mem_fraction_static, args.sglang_max_running_requests) == (
        4096,
        0.8,
        64,
    )
    assert args.sglang_cuda_graph_bs_decode == (1, 2) and not args.pin_rollout_manager_to_head


@pytest.mark.parametrize("hardware", ["H200", "B300"])
@pytest.mark.parametrize("buffer_size", [4 * 1024**3, 8 * 1024**3])
def test_weight_buffer_override_reaches_train_entrypoint(monkeypatch, launcher, hardware, buffer_size):
    tokens, _ = _train_args(monkeypatch, launcher, hardware=hardware, update_weight_buffer_size=buffer_size)
    assert tokens.count("--update-weight-buffer-size") == 1
    assert tokens[tokens.index("--update-weight-buffer-size") + 1] == str(buffer_size)


# The partial on each hardware: GPUs per node -> (TP, PP, CP, EP, layers on each edge stage).
@pytest.mark.parametrize("hardware,num_gpus,layout", [("H200", 8, (4, 2, 1, 4, 2)), ("B300", 4, (2, 2, 1, 2, 2))])
def test_partial_runs_the_four_layer_checkpoint_on_one_node(
    monkeypatch, tmp_path, launcher, hardware, num_gpus, layout
):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    template = tmp_path / "chat.jinja"
    template.write_text("{{ messages }}")
    backend = _run(
        monkeypatch,
        launcher,
        model_name=_PARTIAL,
        hardware=hardware,
        num_gpus_per_node=num_gpus,
        chat_template_path=str(template),
        num_rollout=3,
        rollout_batch_size=4,
        n_samples_per_prompt=4,
    )

    assert backend.commands == [
        "hf download RadixArk/Muse-Spark-Open-last4-partial --local-dir /root/models/Muse-Spark-Open-last4-partial"
    ]
    (conversion,) = backend.conversions
    assert not conversion["multinode"] and conversion["megatron_model_type"] == "muse-spark-open-last4-partial"
    (call,) = backend.trains
    assert call["config"].num_nodes == 1 and call["megatron_model_type"] == "muse-spark-open-last4-partial"

    args = _validate(
        monkeypatch,
        shlex.split(call["train_args"]),
        megatron_model_type="muse-spark-open-last4-partial",
        world_size=num_gpus,
        blackwell=hardware == "B300",
    )
    assert args.hf_checkpoint == "/root/models/Muse-Spark-Open-last4-partial"
    assert args.ref_load == "/root/models/Muse-Spark-Open-last4-partial_torch_dist"
    assert args.num_layers == 4
    tp, pp, cp, ep, edge_layers = layout
    assert (args.tensor_model_parallel_size, args.pipeline_model_parallel_size) == (tp, pp)
    assert (args.context_parallel_size, args.expert_model_parallel_size) == (cp, ep)
    assert args.decoder_first_pipeline_num_layers == args.decoder_last_pipeline_num_layers == edge_layers
    assert (args.actor_num_nodes, args.actor_num_gpus_per_node, args.rollout_num_gpus_per_engine) == (
        1,
        num_gpus,
        num_gpus,
    )
    assert (args.num_rollout, args.rollout_batch_size, args.n_samples_per_prompt) == (3, 4, 4)


@pytest.mark.parametrize("hardware", ["H200", "B300"])
def test_the_full_model_converts_across_the_ray_cluster(monkeypatch, launcher, hardware):
    (conversion,) = _run(monkeypatch, launcher, hardware=hardware).conversions
    assert conversion["multinode"] and conversion["megatron_model_type"] == "muse-spark-open"


def test_vlm_trains_and_evaluates_on_geo3k_imgurl(monkeypatch, launcher):
    backend = _run(monkeypatch, launcher, hardware="H200", vlm=True, eval_interval=5)
    assert backend.datasets == ["chenhegu/geo3k_imgurl"]
    (call,) = backend.trains
    tokens = shlex.split(call["train_args"])
    assert tokens[tokens.index("--prompt-data") + 1] == "/root/datasets/geo3k_imgurl/train.parquet"
    assert tokens[tokens.index("--eval-prompt-data") + 2] == "/root/datasets/geo3k_imgurl/test.parquet"
    assert (tokens[tokens.index("--input-key") + 1], tokens[tokens.index("--label-key") + 1]) == ("problem", "answer")
    assert tokens[tokens.index("--multimodal-keys") + 1] == '{"image": "images"}'
