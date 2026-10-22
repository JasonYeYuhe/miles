from argparse import Namespace
from collections.abc import Callable

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

from miles.backends.training_utils import parallel
from miles.utils.timer import Timer


@pytest.fixture
def native_backend_owner_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel, "_parallel_state", None)
    make_parallel_state()
    monkeypatch.setattr(Timer(), "seq_lens", [], raising=False)


@pytest.fixture
def uneven_policy_batch() -> dict:
    return {
        "advantages": [torch.tensor([1.0]), torch.full((3,), 3.0)],
        "log_probs": [torch.log(torch.full((1,), 0.5)), torch.log(torch.full((3,), 0.5))],
        "rollout_log_probs": [torch.log(torch.full((1,), 0.5)), torch.log(torch.full((3,), 0.5))],
        "unconcat_tokens": [torch.tensor([0, 1]), torch.tensor([0, 1, 0, 1])],
        "response_lengths": [1, 3],
        "total_lengths": [2, 4],
        "loss_masks": [torch.ones(1), torch.ones(3)],
    }


@pytest.fixture
def policy_backend_args() -> Namespace:
    return make_args(entropy_coef=0.0, observe_training_entropy=False, use_tis=True, global_batch_size=2)


def _per_token_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    return torch.sum if calculate_per_token_loss else torch.mean


@pytest.fixture
def cpu_rollout_store(monkeypatch: pytest.MonkeyPatch) -> None:
    from copy import deepcopy
    from types import SimpleNamespace

    from miles.backends.training_utils.data import rollout
    from miles.utils import object_store
    from miles.utils.object_store import ObjectStoreGetResult

    value = {
        "partition": [0],
        "tokens": [list(range(11))],
        "total_lengths": [11],
        "response_lengths": [1],
        "loss_masks": [[1]],
    }
    store = SimpleNamespace(get=lambda ref: ObjectStoreGetResult(deepcopy(value), release_fn=lambda value: None))
    monkeypatch.setattr(object_store, "_INSTANCE", store)
    monkeypatch.setattr(rollout.current_platform, "current_device", lambda: torch.device("cpu"))
