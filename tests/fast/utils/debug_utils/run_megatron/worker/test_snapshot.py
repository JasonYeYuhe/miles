from argparse import Namespace
from pathlib import Path

import pytest

from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
from miles.utils.audit_utils.config_snapshot.normalizer import normalize_record
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.process_identity import TrainProcessIdentity
from miles.utils.debug_utils.run_megatron.worker.script_args import WorkerScriptArgs
from miles.utils.debug_utils.run_megatron.worker.snapshot import configure
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR


class TestStandaloneSnapshot:
    @pytest.mark.parametrize("role", ["actor", "critic"])
    @pytest.mark.parametrize("rank", [0, 3])
    def test_native_arguments_and_worker_identity_reach_storage(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str, rank: int
    ) -> None:
        """Standalone capture preserves native arguments and identifies each training rank."""
        monkeypatch.setattr(ConfigSnapshotDumper, "_state", None)
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path))
        args = Namespace(
            rank=rank, load="/checkpoint", no_load_optim=True, script_config_snapshot_run_uuid="standalone-uuid"
        )
        script = WorkerScriptArgs(
            hf_checkpoint=Path("/hf"),
            token_ids_file=Path("/tokens.json"),
            role=role,
            config_snapshot_name="standalone/run-0000",
            config_snapshot_run_uuid="standalone-uuid",
        )

        configure(args=args, script=script, rank=rank)

        records = ConfigSnapshotStorage(directory=tmp_path).read()
        assert len(records) == 1
        record = records[0]
        assert record.context.name == "standalone/run-0000"
        assert record.context.run_uuid == "standalone-uuid"
        assert record.context.deploy_component == "standalone"
        assert record.context.deploy_instance_id == "default"
        assert record.context.source == TrainProcessIdentity(component=role, cell_index=0, rank_within_cell=rank)
        assert record.point.stage == "process_config"
        assert record.point.index == 0
        assert record.config == {
            "args": {
                "rank": rank,
                "load": "/checkpoint",
                "no_load_optim": True,
                "script_config_snapshot_run_uuid": "standalone-uuid",
            }
        }
        assert vars(args) == {
            "rank": rank,
            "load": "/checkpoint",
            "no_load_optim": True,
            "script_config_snapshot_run_uuid": "standalone-uuid",
        }
        assert normalize_record(record) == {
            "args": {
                "rank": "$RANK",
                "load": "/checkpoint",
                "no_load_optim": True,
                "script_config_snapshot_run_uuid": "$RUN_UUID",
            }
        }

    def test_checkpoint_boundaries_share_the_configured_capture(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checkpoint records retain worker provenance and advance their own stage index."""
        monkeypatch.setattr(ConfigSnapshotDumper, "_state", None)
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path))
        args = Namespace(rank=0, load="/checkpoint")
        script = WorkerScriptArgs(
            hf_checkpoint=Path("/hf"),
            token_ids_file=Path("/tokens.json"),
            config_snapshot_name="standalone/run-0000",
            config_snapshot_run_uuid="standalone-uuid",
        )

        configure(args=args, script=script, rank=0)
        ConfigSnapshotDumper.dump(stage="checkpoint_load", config={"args": args})
        args.load = "/reference"
        ConfigSnapshotDumper.dump(stage="checkpoint_load", config={"args": args})

        records = ConfigSnapshotStorage(directory=tmp_path).read()
        process = [record for record in records if record.point.stage == "process_config"]
        checkpoints = sorted(
            (record for record in records if record.point.stage == "checkpoint_load"),
            key=lambda record: record.point.index,
        )
        assert len(process) == 1
        assert [record.point.index for record in checkpoints] == [0, 1]
        assert [record.config for record in checkpoints] == [
            {"args": {"rank": 0, "load": "/checkpoint"}},
            {"args": {"rank": 0, "load": "/reference"}},
        ]
        assert all(record.context == process[0].context for record in checkpoints)
        assert process[0].config == {"args": {"rank": 0, "load": "/checkpoint"}}

    @pytest.mark.parametrize("disabled_by", ["name", "environment"])
    def test_disabled_capture_clears_the_previous_worker_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, disabled_by: str
    ) -> None:
        """A disabled worker cannot append records to an earlier enabled capture."""
        monkeypatch.setattr(ConfigSnapshotDumper, "_state", None)
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path))
        args = Namespace(rank=0)
        enabled = WorkerScriptArgs(
            hf_checkpoint=Path("/hf"),
            token_ids_file=Path("/tokens.json"),
            config_snapshot_name="enabled/run-0000",
            config_snapshot_run_uuid="enabled-uuid",
        )
        configure(args=args, script=enabled, rank=0)
        before = ConfigSnapshotStorage(directory=tmp_path).read()
        if disabled_by == "environment":
            monkeypatch.delenv(SNAPSHOT_RECORD_DIR_ENV_VAR)
        disabled = WorkerScriptArgs(
            hf_checkpoint=Path("/hf"),
            token_ids_file=Path("/tokens.json"),
            config_snapshot_name=None if disabled_by == "name" else "disabled/run-0000",
        )

        configure(args=args, script=disabled, rank=0)
        ConfigSnapshotDumper.dump(stage="checkpoint_load", config={"args": args})

        assert ConfigSnapshotStorage(directory=tmp_path).read() == before
        assert ConfigSnapshotDumper._state is None

    @pytest.mark.parametrize("run_uuid", [None, ""])
    def test_enabled_capture_rejects_a_missing_run_uuid(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run_uuid: str | None
    ) -> None:
        """Enabled standalone captures require a shared nonempty run identity."""
        monkeypatch.setattr(ConfigSnapshotDumper, "_state", None)
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path))
        script = WorkerScriptArgs(
            hf_checkpoint=Path("/hf"),
            token_ids_file=Path("/tokens.json"),
            config_snapshot_name="standalone/run-0000",
            config_snapshot_run_uuid=run_uuid,
        )

        with pytest.raises(AssertionError, match="nonempty run UUID"):
            configure(args=Namespace(rank=0), script=script, rank=0)

        assert ConfigSnapshotStorage(directory=tmp_path).read() == []
