"""Check that an installed Miles wheel can start an FSDP or torchtitan run without Megatron.

Megatron-LM, Megatron Bridge, mbridge, TransformerEngine and apex are made to look uninstalled, so
the check fails if any module the FSDP or torchtitan path loads at import time reaches one of them,
even on a machine where they are installed. Run it from outside the checkout so ``miles`` resolves
to the installed wheel:

    python tests/ci/wheel_smoke.py
"""

import importlib
import importlib.metadata
import sys
import sysconfig
from pathlib import Path

MEGATRON_ONLY = ("megatron", "mbridge", "transformer_engine", "apex")
BACKENDS = ("fsdp", "torchtitan")
COMMANDS = {
    "miles-train": "miles.entrypoints.train",
    "miles-train-async": "miles.entrypoints.train_async",
}


def _check_miles_is_installed() -> None:
    import miles

    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    origin = Path(miles.__file__).resolve()
    assert origin.is_relative_to(site_packages), f"miles resolved to {origin}, not the wheel in {site_packages}"


def _check_commands() -> None:
    scripts = Path(sysconfig.get_paths()["scripts"])
    entry_points = {ep.name: ep for ep in importlib.metadata.entry_points(group="console_scripts")}
    for name, module in COMMANDS.items():
        assert (scripts / name).is_file(), f"{name} is not installed in {scripts}"
        entry_point = entry_points[name]
        assert entry_point.value == f"{module}:main", f"{name} runs {entry_point.value}"
        entry_point.load()


def _parse_args(backend: str) -> None:
    from miles.utils.arguments import parse_args

    sys.argv = ["miles-train", "--train-backend", backend]
    sys.argv += ["--num-rollout", "1", "--rollout-batch-size", "1", "--rollout-max-response-len", "128"]
    args = parse_args()
    assert args.train_backend == backend


def _check_chat_templates() -> None:
    from miles.utils.chat_template_utils import TEMPLATE_DIR

    assert sorted(TEMPLATE_DIR.glob("*.jinja")), f"no chat templates shipped in {TEMPLATE_DIR}"


def main() -> None:
    # A None entry is how a missing package looks: imports raise ModuleNotFoundError and
    # importlib.util.find_spec probes return None, even where the package is installed.
    for name in MEGATRON_ONLY:
        sys.modules[name] = None

    _check_miles_is_installed()
    _check_commands()
    for backend in BACKENDS:
        importlib.import_module(f"miles.backends.{backend}_utils.actor")
        _parse_args(backend)
    _check_chat_templates()

    print(f"wheel smoke passed: {', '.join(COMMANDS)} with {', '.join(BACKENDS)} and no Megatron")


if __name__ == "__main__":
    main()
