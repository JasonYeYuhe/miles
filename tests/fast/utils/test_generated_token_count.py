from miles.utils.types import Sample


def test_generation_count_survives_serialization_and_resets_for_retry() -> None:
    sample = Sample(response_length=2, generated_token_count=10, loss_mask=[0, 0])
    restored = Sample.from_dict(sample.to_dict())
    assert restored.generated_tokens == 10
    restored.reset_for_retry()
    assert restored.generated_token_count is None
    assert restored.generated_tokens == 0
