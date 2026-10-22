import inspect

import pytest

from miles.backends.megatron_utils.checkpoint_request import MegatronCkptLoadConfig
from miles.backends.torchtitan_utils.actor import TorchtitanTrainRayActor


class TestTorchtitanCheckpointRequestRejection:
    def test_megatron_load_request_is_rejected_before_torchtitan_initialization(self) -> None:
        """TorchTitan cannot silently accept Megatron-only runtime loading inputs."""
        actor = TorchtitanTrainRayActor.__new__(TorchtitanTrainRayActor)
        original = vars(actor).copy()

        with pytest.raises(AssertionError, match="does not support checkpoint healing"):
            inspect.unwrap(TorchtitanTrainRayActor.init)(
                actor,
                args=None,
                role="actor",
                ckpt_load_config=MegatronCkptLoadConfig.for_peer_recovery(),
                indep_dp_info=None,
                indep_dp_store_addr=None,
            )

        assert vars(actor) == original
