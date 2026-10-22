"""Check that an installed Miles wheel can start an FSDP or torchtitan run without Megatron.

Megatron-LM, Megatron Bridge, mbridge, TransformerEngine and apex are made to look uninstalled, so
the check fails if any module the FSDP or torchtitan path loads at import time reaches one of them,
even on a machine where they are installed. The entry scripts are not part of the wheel, so they
are loaded from the directory given on the command line. Run it from outside the checkout so
``miles`` resolves to the installed wheel:

    python tests/ci/wheel_smoke.py <checkout>
"""

import importlib
import importlib.util
import sys
import sysconfig
from pathlib import Path

MEGATRON_ONLY = ("megatron", "mbridge", "transformer_engine", "apex")
BACKENDS = ("fsdp", "torchtitan")
ENTRY_SCRIPTS = ("train.py", "train_async.py")


def _check_miles_is_installed() -> None:
    import miles

    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    origin = Path(miles.__file__).resolve()
    assert origin.is_relative_to(site_packages), f"miles resolved to {origin}, not the wheel in {site_packages}"


def _load_entry_scripts(checkout: Path) -> None:
    for script in ENTRY_SCRIPTS:
        spec = importlib.util.spec_from_file_location(f"_wheel_smoke_{Path(script).stem}", checkout / script)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))


def _parse_args(backend: str) -> None:
    from miles.utils.arguments import parse_args

    sys.argv = ["train.py", "--train-backend", backend]
    sys.argv += ["--num-rollout", "1", "--rollout-batch-size", "1", "--rollout-max-response-len", "128"]
    args = parse_args()
    assert args.train_backend == backend


def _check_chat_templates() -> None:
    from miles.utils.chat_template_utils import TEMPLATE_DIR

    assert sorted(TEMPLATE_DIR.glob("*.jinja")), f"no chat templates shipped in {TEMPLATE_DIR}"


def main() -> None:
    checkout = Path(sys.argv[1]).resolve()
    # A None entry is how a missing package looks: imports raise ModuleNotFoundError and
    # importlib.util.find_spec probes return None, even where the package is installed.
    for name in MEGATRON_ONLY:
        sys.modules[name] = None

    _check_miles_is_installed()
    _load_entry_scripts(checkout)
    for backend in BACKENDS:
        importlib.import_module(f"miles.backends.{backend}_utils.actor")
        _parse_args(backend)
    _check_chat_templates()

    print(f"wheel smoke passed: {', '.join(BACKENDS)} without Megatron")


if __name__ == "__main__":
    main()
