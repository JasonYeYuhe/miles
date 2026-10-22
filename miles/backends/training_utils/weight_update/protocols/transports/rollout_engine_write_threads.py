from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial

import torch

from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import (
    RemoteShard,
    assert_tensors_fit,
)
from miles.utils.test_utils.fault_injector.actions.base import FaultHookContext
from miles.utils.test_utils.fault_injector.controller import fault_hook_controller


class RolloutEngineWriteThreads:
    """One write thread per rollout engine, so a stuck engine holds up only its own writes."""

    def __init__(self, thread_name_prefix: str) -> None:
        self._thread_name_prefix = thread_name_prefix
        self._executors_by_rollout_engine_ind: dict[int, ThreadPoolExecutor] = {}

    def restart(self) -> None:
        """Called at connect: a rollout engine index now names a new engine, which must not queue behind the writes
        of the one it replaced."""
        for executor in self._executors_by_rollout_engine_ind.values():
            executor.shutdown(wait=False, cancel_futures=True)
        self._executors_by_rollout_engine_ind = {}

    def write_each_shard(
        self,
        remote_shards: Sequence[RemoteShard],
        tensors_by_name: dict[str, torch.Tensor],
        *,
        run_write: Callable[[RemoteShard, Callable[[], None]], None],
        write_shard: Callable[..., None],
    ) -> list[Future[None]]:
        """Runs `write_shard(remote_shard, tensors_by_name, context=...)` through `run_write` on each shard's rollout
        engine thread, with the caller's fault hook context; returns one future per shard. Raises before writing
        anything if a shard lacks one of the weights or holds it in a different number of bytes."""
        for remote_shard in remote_shards:
            assert_tensors_fit(remote_shard, tensors_by_name)
        context: FaultHookContext = fault_hook_controller.current_context()
        return [
            self._get_or_create_executor(remote_shard.rollout_engine_ind).submit(
                run_write, remote_shard, partial(write_shard, remote_shard, tensors_by_name, context=context)
            )
            for remote_shard in remote_shards
        ]

    def _get_or_create_executor(self, rollout_engine_ind: int) -> ThreadPoolExecutor:
        if rollout_engine_ind not in self._executors_by_rollout_engine_ind:
            self._executors_by_rollout_engine_ind[rollout_engine_ind] = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"{self._thread_name_prefix}-rollout-engine-{rollout_engine_ind}"
            )
        return self._executors_by_rollout_engine_ind[rollout_engine_ind]
