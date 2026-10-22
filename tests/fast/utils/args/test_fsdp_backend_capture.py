from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

from miles.utils.args.runtime import AllConfig


_BACKEND_FIELDS = frozenset(
    {
        "bf16",
        "calculate_per_token_loss",
        "ckpt_step",
        "clip_grad",
        "disable_grad_buffers_cpu_backup",
        "disable_param_buffers_cpu_backup",
        "finetune",
        "load",
        "mtp_detach_heads",
        "mtp_loss_scaling_factor",
        "mtp_num_layers",
        "no_load_optim",
        "no_load_rng",
        "no_save_optim",
        "rank",
        "world_size",
    }
)


class TestFsdpBackendCapture:
    def test_native_backend_fields_survive_real_parsing_and_json_transport(self) -> None:
        """Native backend capture retains shared and normalized fields without swallowing launcher fields."""
        config = parse_fsdp_test_config("--clip-grad", "0.75")
        restored = AllConfig.model_validate_json(config.model_dump_json())
        backend = vars(restored.raw_fsdp)

        assert _BACKEND_FIELDS.issubset(backend)
        assert backend["clip_grad"] == 0.75
        assert backend["rank"] == 0
        assert backend["world_size"] == 1
        assert backend == vars(config.raw_fsdp)
        assert not {"actor_num_nodes", "actor_num_gpus_per_node", "rollout_function_path"}.intersection(backend)
