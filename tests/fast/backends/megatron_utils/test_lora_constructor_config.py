import pytest
from tests.fast.fixtures.args_fixtures import make_trainer_args

from miles.backends.megatron_utils.lora.slots import create_multi_lora_instance
from miles.backends.megatron_utils.lora.utils import create_lora_instance


class TestLoraConstructorConfig:
    @pytest.mark.parametrize("kind,expected_type", [("lora", "LoRA"), ("canonical_lora", "CanonicalLoRA")])
    def test_the_real_adapter_config_preserves_explicit_rank_dropout_and_initialization(
        self, kind: str, expected_type: str
    ) -> None:
        """The selected Bridge adapter receives the requested optimizer-independent LoRA settings."""
        args = make_trainer_args(
            lora_type=kind,
            lora_rank=4,
            lora_alpha=12,
            lora_dropout=0.25,
            lora_A_init_method="normal",
            lora_B_init_method="zero",
            experts_shared_outer_loras=False,
        )

        adapter = create_lora_instance(args, target_modules=["linear_qkv"])

        assert type(adapter).__name__ == expected_type
        assert adapter.dim == 4
        assert adapter.alpha == 12
        assert adapter.dropout == 0.25
        assert adapter.lora_A_init_method == "normal"
        assert adapter.lora_B_init_method == "zero"
        assert adapter.target_modules == ["linear_qkv"]

    def test_multi_adapter_config_preserves_slot_count_and_dropout(self) -> None:
        """Multi-LoRA construction preserves explicit slot count and initialization settings."""
        args = make_trainer_args(
            multi_lora_n_adapters=3,
            lora_rank=4,
            lora_alpha=12,
            lora_dropout=0.25,
            lora_A_init_method="normal",
            lora_B_init_method="zero",
        )

        adapter = create_multi_lora_instance(args, target_modules=["linear_qkv"])

        assert adapter.n_adapters == 3
        assert adapter.dim == 4
        assert adapter.dropout == 0.25
        assert adapter.lora_B_init_method == "zero"

    def test_canonical_adapter_rejects_the_shared_expert_mode(self) -> None:
        """Canonical LoRA cannot silently accept the standard adapter's shared-expert layout."""
        args = make_trainer_args(
            lora_type="canonical_lora", experts_shared_outer_loras=True, lora_B_init_method="zero"
        )

        with pytest.raises(AssertionError, match="requires the standard LoRA adapter type"):
            create_lora_instance(args, target_modules=["linear_qkv"])
