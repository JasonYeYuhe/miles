from argparse import Namespace

import pytest

from miles.ray.rollout.train_data_conversion import _post_process_rewards, convert_samples_to_train_data
from miles.rollout.filter_hub.common_filters import apply_reward_nonzero_std_filter
from miles.rollout.reward_shaping import apply_length_penalty, reward_group_segments, validate_length_penalty_args
from miles.utils.types import Sample


def _args(**overrides) -> Namespace:
    return Namespace(
        **{
            "length_penalty_coeff": 1.0,
            "length_penalty_min_tokens": 1,
            "reward_key": None,
            "n_samples_per_prompt": 4,
            "rollout_batch_size": 2,
            "advantage_estimator": "grpo",
            "rewards_normalization": True,
            "grpo_std_normalization": False,
            "enable_sample_ownership_checker": False,
            "rollout_top_logprobs_num": 0,
            "use_dynamic_global_batch_size": False,
            **overrides,
        }
    )


def _shape(args: Namespace, samples: list[Sample]) -> list[float]:
    rewards = [sample.get_reward_value(args) for sample in samples]
    return apply_length_penalty(args, samples, rewards, reward_group_segments(args, samples))


@pytest.mark.parametrize("coeff, expected", [(0.0, 1.0), (0.5, 2**-0.5), (1.0, 0.5), (2.0, 0.25)])
def test_correct_only_and_configurable_exponent(coeff: float, expected: float) -> None:
    samples = [
        Sample(group_index=0, response_length=100, reward=1.0),
        Sample(group_index=0, response_length=200, reward=1.0),
        Sample(group_index=0, response_length=1, reward=0.0),
        Sample(group_index=0, response_length=1000, reward=-0.5),
        Sample(group_index=0, response_length=1000, reward=0.9),
    ]
    assert _shape(_args(length_penalty_coeff=coeff), samples) == pytest.approx([1, expected, 0, -0.5, 0.9])
    assert [sample.reward for sample in samples] == [1.0, 1.0, 0.0, -0.5, 0.9]


def test_minimum_protects_zero_and_short_trajectories() -> None:
    samples = [Sample(group_index=0, response_length=length, reward=1.0) for length in [0, 20, 100, 200]]
    assert _shape(_args(length_penalty_min_tokens=100), samples) == [1, 1, 1, 0.5]
    assert _shape(_args(), [samples[0], samples[3]]) == [1, 0.005]


def test_ties_single_correct_and_no_correct() -> None:
    samples = [
        Sample(group_index=0, response_length=10, reward=1),
        Sample(group_index=1, response_length=1, reward=0),
        Sample(group_index=0, response_length=10, reward=1),
        Sample(group_index=1, response_length=100, reward=-1),
        Sample(group_index=2, response_length=200, reward=1),
    ]
    assert _shape(_args(), samples) == [1, 0, 1, -1, 1]
    assert _shape(_args(), []) == []


def test_interleaved_prompt_groups_and_rollout_segments() -> None:
    samples = [
        Sample(group_index=0, rollout_id=10, response_length=40, reward=1),
        Sample(group_index=1, rollout_id=20, response_length=200, reward=1),
        Sample(group_index=0, rollout_id=11, response_length=200, reward=1),
        Sample(group_index=0, rollout_id=10, response_length=60, reward=1),
        Sample(group_index=1, rollout_id=21, response_length=400, reward=1),
    ]
    assert _shape(_args(), samples) == [1, 1, 0.5, 1, 0.5]


def test_explicit_prompt_boundaries_override_indices() -> None:
    samples = [Sample(group_index=0, response_length=length, reward=1) for length in [10, 20, 100, 400]]
    rewards = apply_length_penalty(_args(), samples, [1] * 4, reward_group_segments(_args(), samples, [2, 2]))
    assert rewards == [1, 0.5, 1, 0.25]


def test_explicit_generation_count_ignores_loss_mask_and_observations() -> None:
    samples = [
        Sample(group_index=0, response_length=1000, generated_token_count=100, loss_mask=[0] * 1000, reward=1),
        Sample(group_index=0, response_length=2000, generated_token_count=200, loss_mask=[1] * 2000, reward=1),
    ]
    assert _shape(_args(), samples) == [1, 0.5]

    # Legacy/custom multi-turn producers can use a mask that marks only generated tokens.
    samples[0].generated_token_count = None
    samples[0].loss_mask = [1] * 100 + [0] * 900
    assert _shape(_args(), samples) == [1, 0.5]


def test_generation_count_survives_serialization_and_resets_for_retry() -> None:
    sample = Sample(response_length=2, generated_token_count=10, loss_mask=[0, 0])
    restored = Sample.from_dict(sample.to_dict())
    assert restored.generated_tokens == 10
    restored.reset_for_retry()
    assert restored.generated_token_count is None
    assert restored.generated_tokens == 0


def test_dict_reward_and_inconsistent_siblings() -> None:
    samples = [
        Sample(group_index=0, rollout_id=1, response_length=10, reward={"accuracy": 1}),
        Sample(group_index=0, rollout_id=2, response_length=20, reward={"accuracy": 1}),
    ]
    assert _shape(_args(reward_key="accuracy"), samples) == [1, 0.5]
    samples[1].rollout_id = 1
    samples[1].reward["accuracy"] = 0
    with pytest.raises(ValueError, match="share one reward"):
        _shape(_args(reward_key="accuracy"), samples)


def test_dynamic_filter_keeps_all_correct_groups_with_length_signal() -> None:
    samples = [Sample(group_index=0, index=i, response_length=length, reward=1) for i, length in enumerate([10, 20])]
    assert apply_reward_nonzero_std_filter(_args(), samples).keep
    assert not apply_reward_nonzero_std_filter(_args(length_penalty_coeff=0), samples).keep
    samples[1].response_length = 10
    assert not apply_reward_nonzero_std_filter(_args(), samples).keep


@pytest.mark.parametrize("normalize", [False, True])
def test_shaping_reaches_training_data_before_normalization(normalize: bool) -> None:
    samples = [
        Sample(index=i, group_index=0, tokens=[0] * length, response_length=length, reward=1.0)
        for i, length in enumerate([10, 20])
    ]
    out = convert_samples_to_train_data(
        _args(rewards_normalization=normalize),
        samples,
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )
    assert out["raw_reward"] == [1.0, 1.0]
    assert out["rewards"] == pytest.approx([0.25, -0.25] if normalize else [1.0, 0.5])
    assert [sample.reward for sample in samples] == [1.0, 1.0]


@pytest.mark.parametrize("std_normalize", [False, True])
def test_normalization_counts_each_rollout_once(std_normalize: bool) -> None:
    samples = [
        Sample(index=0, rollout_id=10, group_index=0, response_length=4, reward=1.0),
        Sample(index=1, rollout_id=10, group_index=0, response_length=6, reward=1.0),
        Sample(index=2, rollout_id=11, group_index=0, response_length=20, reward=1.0),
    ]
    raw, rewards = _post_process_rewards(_args(grpo_std_normalization=std_normalize), samples, None)
    positive = 0.25 / (0.5 / 2**0.5 + 1e-6) if std_normalize else 0.25
    assert raw == [1.0, 1.0, 1.0]
    assert rewards == pytest.approx([positive, positive, -positive])


def test_zero_coefficient_disables_shaping_in_training() -> None:
    samples = [Sample(group_index=0, index=i, response_length=length, reward=1.0) for i, length in enumerate([0, 100])]
    raw, rewards = _post_process_rewards(_args(length_penalty_coeff=0, rewards_normalization=False), samples, None)
    assert raw == rewards == [1.0, 1.0]


@pytest.mark.parametrize("coeff", [-1, float("inf"), float("nan")])
def test_invalid_coefficients(coeff: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_length_penalty_args(_args(length_penalty_coeff=coeff))


@pytest.mark.parametrize("minimum", [0, -1, 1.5, True])
def test_invalid_minimum(minimum: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_length_penalty_args(_args(length_penalty_min_tokens=minimum))


@pytest.mark.parametrize("field", ["custom_reward_post_process_path", "custom_convert_samples_to_train_data_path"])
def test_custom_overrides_cannot_silently_bypass_shaping(field: str) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_length_penalty_args(_args(**{field: "custom.function"}))
    validate_length_penalty_args(_args(length_penalty_coeff=0, **{field: "custom.function"}))
