from __future__ import annotations

import functools
from typing import Any

from miles.utils.hf_utils.config import read_checkpoint_model_type

_MODEL_TYPES = ("inkling_mm_model", "inkling_mm_model")


def is_inkling(tokenizer: Any) -> bool:
    return read_checkpoint_model_type(getattr(tokenizer, "name_or_path", "")) in _MODEL_TYPES


def is_inkling_checkpoint(name_or_path: str) -> bool:
    return read_checkpoint_model_type(name_or_path) in _MODEL_TYPES


@functools.cache
def fixed_chat_template() -> str:
    """The bundled Inkling chat template (the checkpoint's, with the upstream fixes)."""
    # imported lazily: tito_tokenizer imports template.py, which imports this module
    from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR

    return (TEMPLATE_DIR / "inkling_fixed.jinja").read_text(encoding="utf-8")
