import os
from argparse import Namespace

from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
from miles.utils.audit_utils.process_identity import TrainProcessIdentity
from miles.utils.debug_utils.run_megatron.worker.script_args import WorkerScriptArgs
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR


def configure(*, args: Namespace, script: WorkerScriptArgs, rank: int) -> None:
    enabled = bool(script.config_snapshot_name and os.environ.get(SNAPSHOT_RECORD_DIR_ENV_VAR))
    if enabled:
        assert script.config_snapshot_run_uuid, "Standalone snapshot capture requires a nonempty run UUID"

    metadata = Namespace(
        ci_disable_config_snapshot=not enabled,
        config_snapshot_name=script.config_snapshot_name,
        deploy_component="standalone",
        deploy_instance_id="default",
        run_uuid=script.config_snapshot_run_uuid,
    )
    ConfigSnapshotDumper.configure(
        args=metadata,
        source=TrainProcessIdentity(component=script.role, cell_index=0, rank_within_cell=rank),
    )
    ConfigSnapshotDumper.dump(stage="process_config", config={"args": args})
