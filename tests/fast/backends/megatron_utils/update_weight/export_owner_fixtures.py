from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state

from miles.backends.megatron_utils import named_weights
from miles.backends.megatron_utils.update_weight import hf_weight_iterator_direct
from miles.backends.training_utils import parallel
from miles.backends.training_utils.parallel import GroupInfo
from miles.utils import distributed_utils


@pytest.fixture
def duplicate_param_model(monkeypatch: pytest.MonkeyPatch) -> torch.nn.Module:
    monkeypatch.setattr(parallel, "_parallel_state", None)
    state = make_parallel_state()
    state.pp = GroupInfo(rank=1, size=2, group=object())
    monkeypatch.setattr(named_weights, "get_transformer_layer_offset", lambda config, vp_stage=0: 0)
    monkeypatch.setattr(hf_weight_iterator_direct.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(hf_weight_iterator_direct.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", object())

    def gather(*, obj: object, object_list: list, group: object) -> None:
        if isinstance(obj, tuple):
            rank, infos = obj
            object_list[:] = [(rank, infos), (0, {name: replace(info, src_rank=0) for name, info in infos.items()})]
        else:
            object_list[:] = [obj, obj]

    monkeypatch.setattr(hf_weight_iterator_direct.dist, "all_gather_object", gather)
    model = torch.nn.Linear(2, 2, bias=False)
    model.config = SimpleNamespace()
    return model
