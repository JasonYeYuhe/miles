from scripts.run_muse_spark_open import ScriptArgs, execute, prepare
from tests.ci.ci_register import register_cuda_ci

# Smoke test of Muse Spark Open on the last-four-layer partial on one node: download and torch_dist conversion,
# then three rollout/train/sync cycles with routing replay, the --ci-test log-prob and KL checks and the
# post-update weight check. The partial is not a language model, so a math reward would be all zeros and no
# update would move the weights; a fixed pseudo-random reward keeps them moving.

register_cuda_ci(
    est_time=2400,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "model-scripts"],
    hardware=["hopper"],
    num_gpus=8,
)

# The text recipe's colocated engines zero the omni towers at startup and never receive them again.
_FROZEN_TOWERS = "model.vision_tower. model.vision_adapter. model.vision_projection. model.audio_tower."


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="Muse-Spark-Open-last4-partial",
        hardware="H200",
        num_gpus_per_node=8,
        num_rollout=3,
        rollout_batch_size=4,
        n_samples_per_prompt=4,
        rollout_max_response_len=1024,
        rollout_max_context_len=4096,
        rm_type="deterministic_random",
        extra_args=f"--ci-test --check-weight-update-skip-list {_FROZEN_TOWERS} ",
    )


if __name__ == "__main__":
    args = _args()
    prepare(args)
    execute(args)
