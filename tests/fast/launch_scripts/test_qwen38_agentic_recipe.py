import json
import shlex

import pytest

from tests.fast.launch_scripts.py_harness import (
    FROZEN_RUN_ID,
    format_recording,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot

from miles.utils.args.runtime import AllConfig

_LAUNCHER = "examples/experimental/harbor/run_qwen38_27b.py"
_DATA = "examples/experimental/harbor/prepare_terminal_bench.py"


def _module():
    return import_launch_script(REPO_ROOT / _LAUNCHER)


def test_default_recipe_has_separate_trainers_and_manual_min_load_rollouts():
    module = _module()
    args = module.ScriptArgs()
    argv = shlex.split(module._build_train_args(args))
    for flag, expected in {
        "--actor-num-nodes": "2",
        "--rollout-num-gpus": "48",
        "--global-batch-size": "256",
        "--save-interval": "25",
        "--tito-model": "qwen38small",
        "--sglang-router-policy": "manual",
        "--router-assignment-mode": "min_load",
        "--max-seq-len": "131072",
        "--rollout-max-response-len": "49152",
    }.items():
        assert argv[argv.index(flag) + 1] == expected
    assert "--fully-async" in argv and "--use-tis" in argv
    assert "--use-rollout-logprobs" not in argv
    assert "--colocate" not in argv and "--rollout-function-path" not in argv


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_nodes": 2},
        {"trainer_nodes": 0},
        {"num_gpus_per_node": 3},
        {"rollout_max_response_len": 131072},
        {"save_interval": 0},
    ],
)
def test_reject_invalid_topology_and_budgets(overrides):
    with pytest.raises(ValueError):
        _module().ScriptArgs(**overrides)


def test_runtime_budget_wins_over_inherited_environment(monkeypatch):
    module = _module()
    monkeypatch.setattr(
        module._HARBOR,
        "harbor_env_vars",
        lambda args: {
            "HARBOR_TERMINUS_2_ENABLE_SUMMARIZE": "true",
            "AGENT_MAX_OUTPUT_TOKENS": "1",
        },
    )
    env = module._runtime_env(module.ScriptArgs())
    assert env["HARBOR_TERMINUS_2_ENABLE_SUMMARIZE"] == "false"
    assert env["AGENT_MAX_OUTPUT_TOKENS"] == "49152"
    assert env["AGENT_MAX_INPUT_TOKENS"] == "81920"
    assert env["HARBOR_MAX_SEQ_LEN"] == "131072"


@pytest.mark.parametrize("entrypoint", ["prepare", "execute"])
def test_entrypoint_command_snapshot(entrypoint, monkeypatch, tmp_path):
    freeze_environment(monkeypatch)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    recording = install_command_recorder(monkeypatch)
    module = _module()
    # Shared provider provisioning is tested separately. Recording a launch
    # must not need a real Harbor installation or a sandbox API credential.
    monkeypatch.setattr(
        module._HARBOR,
        "harbor_env_vars",
        lambda args: {
            "HARBOR_ENV_TYPE": args.harbor_env_type,
            "HARBOR_TASKS_DIR": args.harbor_tasks_dir,
            "HARBOR_TRIALS_DIR": args.harbor_trials_dir,
            "AGENT_MODEL_NAME": args.agent_model_name,
            "AGENT_TIMEOUT": str(args.agent_timeout),
        },
    )
    args = module.ScriptArgs(model_dir=str(tmp_path / "models"), output_dir="/scratch/runs", run_id=FROZEN_RUN_ID)
    getattr(module, entrypoint)(args)
    assert recording.commands
    assert not any("pkill" in command or "ray stop" in command for command in recording.commands)
    # The generic launch_scripts tree owns only its three discovered families;
    # this separately tested Harbor example keeps its goldens outside that tree.
    snapshot = REPO_ROOT / "tests/snapshots/harbor_qwen38" / f"{entrypoint}.txt"
    formatted = format_recording(recording, sandbox=tmp_path)
    assert_matches_snapshot(snapshot, "\n".join(line.rstrip() for line in formatted.split("\n")), entrypoint)


def test_dry_run_does_not_convert_or_submit(monkeypatch, capsys):
    module = _module()

    def unexpected(*args, **kwargs):
        raise AssertionError("dry-run reached a side effect")

    monkeypatch.setattr(module, "prepare", unexpected)
    monkeypatch.setattr(module.ScriptArgs, "create_backend", unexpected)
    module.main(module.ScriptArgs(dry_run=True))
    assert json.loads(capsys.readouterr().out)["train_script"] == "train_async.py"


def test_cli_resolves_inherited_dataclass_types(monkeypatch):
    module = _module()
    monkeypatch.setattr("sys.argv", ["recipe", "--dry_run", "--data_dir", "/tmp/tasks", "--num_nodes", "4"])
    args = module._parse_args()
    assert args.dry_run and args.data_dir == "/tmp/tasks" and args.num_nodes == 4
    assert args.cluster_backend.value == "ray"


def test_real_launch_requires_external_ray_without_cleanup(monkeypatch):
    module = _module()
    monkeypatch.delenv("MILES_SCRIPT_EXTERNAL_RAY", raising=False)
    with pytest.raises(ValueError, match="host-wide"):
        module.main(module.ScriptArgs())


def test_miles_parser_accepts_recipe_specific_flags():
    module = _module()
    parser = argparse.ArgumentParser()
    AllConfig.add_arguments(parser)
    parsed, unknown = parser.parse_known_args(shlex.split(module._build_train_args(module.ScriptArgs())))
    # Megatron adds its own optimizer/model-parallel flags in the real entrypoint.
    # Miles must recognize every agentic, async and SGLang flag itself.
    assert not [
        flag for flag in unknown if flag.startswith(("--sglang", "--router", "--async", "--session", "--tito"))
    ]
    assert parsed.fully_async and parsed.use_tis
    assert parsed.use_session_server in (True, "v1")


def test_task_conversion_uses_ids_and_never_solution_text(tmp_path):
    module = import_launch_script(REPO_ROOT / _DATA)
    tasks = tmp_path / "tasks"
    for name in ("a", "b"):
        task = tasks / name
        task.mkdir(parents=True)
        (task / "task.toml").write_text('version = "1.0"\n')
        (task / "instruction.md").write_text("Solve the task.")
        (task / "solution").mkdir()
        (task / "solution/solve.sh").write_text("SECRET_ORACLE_ANSWER")
    output = tmp_path / "train.jsonl"
    assert module.prepare_data(tasks, output, ["b"]) == 1
    assert json.loads(output.read_text()) == {
        "prompt": "a",
        "metadata": {"instance_id": "a", "agent_name": "terminus-2"},
    }
    assert "SECRET_ORACLE_ANSWER" not in output.read_text()
    with pytest.raises(FileExistsError):
        module.prepare_data(tasks, output, [])
    with pytest.raises(ValueError, match="Unknown excluded"):
        module.prepare_data(tasks, tmp_path / "other.jsonl", ["typo"])


import argparse
