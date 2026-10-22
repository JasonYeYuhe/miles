import sys
from collections.abc import Callable
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def raw_mbridge_loader(monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], list[set[str]]]:
    def configure(fail: bool) -> list[set[str]]:
        observed: list[set[str]] = []

        def load_weights(models: list[torch.nn.Module], path: str, *, memory_efficient: bool) -> None:
            for model in models:
                observed.append(set(model.state_dict()))
                model.weight.data.fill_(5)
            if fail:
                raise RuntimeError("HF loading failed")

        bridge = Mock(load_weights=load_weights)
        mbridge = ModuleType("mbridge")
        mbridge.AutoBridge = Mock(from_pretrained=Mock(return_value=bridge))
        monkeypatch.setitem(sys.modules, "mbridge", mbridge)
        monkeypatch.setitem(sys.modules, "miles_plugins.mbridge", ModuleType("miles_plugins.mbridge"))
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        return observed

    return configure


@pytest.fixture
def bridge_sdk(monkeypatch: pytest.MonkeyPatch) -> Mock:
    sdk = Mock()
    module = ModuleType("megatron.bridge")
    module.AutoBridge = sdk
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return sdk


@pytest.fixture
def transformer_engine_quantizer(monkeypatch: pytest.MonkeyPatch) -> Mock:
    sdk = Mock()
    engine = ModuleType("transformer_engine")
    engine.pytorch = ModuleType("transformer_engine.pytorch")
    engine.pytorch.MXFP8Quantizer = sdk
    constants = ModuleType("transformer_engine.pytorch.constants")
    constants.TE_DType = {torch.float8_e4m3fn: object()}
    engine.pytorch.constants = constants
    for module in (engine, engine.pytorch, constants):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return sdk
