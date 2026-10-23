from types import SimpleNamespace

from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload, update_sample_from_response
from miles.utils.types import Sample


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,
        rollout_top_logprobs_num=0,
        lora_rank=0,
        lora_adapter_path=None,
    )


class TestComputeRequestPayloadExtraKey:
    def test_no_extra_key_leaves_the_payload_without_one(self):
        """Without an extra key the payload has no extra_key field at all."""
        payload, halt_status = compute_request_payload(_make_args(), input_ids=[1, 2], sampling_params={})

        assert halt_status is None
        assert "extra_key" not in payload

    def test_an_extra_key_reaches_the_payload_verbatim(self):
        """The extra key is forwarded unchanged to the engine request."""
        payload, halt_status = compute_request_payload(
            _make_args(), input_ids=[1, 2], sampling_params={}, kv_cache_namespace="train:-:7"
        )

        assert halt_status is None
        assert payload["extra_key"] == "train:-:7"


async def test_generation_count_accumulates_across_masked_partial_rollouts() -> None:
    args = _make_args()
    args.sglang_speculative_algorithm = None
    sample = Sample(tokens=[1, 10, 11, 12], response_length=3, generated_token_count=3, loss_mask=[0, 0, 0])
    output = {
        "text": "next",
        "meta_info": {"output_token_logprobs": [(-0.1, 20), (-0.2, 21)], "finish_reason": {"type": "stop"}},
    }
    await update_sample_from_response(args, sample, {"input_ids": [1]}, output, update_loss_mask=True)
    assert sample.generated_tokens == 5
    assert sample.effective_response_length == 2
    sample.loss_mask = [0] * sample.response_length
    await update_sample_from_response(args, sample, {"input_ids": [1]}, output, update_loss_mask=True)
    assert sample.generated_tokens == 7
    assert sample.effective_response_length == 2
