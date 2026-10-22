"""Last-four-layer Muse Spark Open text partial (RadixArk/Muse-Spark-Open-last4-partial)."""

from model_args_utils import load_sibling_model_args


def model_args() -> str:
    return load_sibling_model_args(__file__, "muse-spark-open", num_layers=4)
