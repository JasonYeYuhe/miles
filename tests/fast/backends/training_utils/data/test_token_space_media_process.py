import torch
from sglang.srt.multimodal.modality import Modality

from miles.backends.training_utils.data import token_space_media_process


class _TextMropeStrategy:
    @staticmethod
    def compute_mrope_positions(input_ids, media_features):
        return torch.arange(len(input_ids)).expand(3, -1), None


class _RecordingProcessor:
    strategy = _TextMropeStrategy()

    def __init__(self):
        self.calls = []

    async def process(self, **kwargs):
        self.calls.append(kwargs)
        num_tokens = len(kwargs["input_ids"])
        return {
            "input_ids": torch.tensor([kwargs["input_ids"]]),
            "attention_mask": torch.ones(1, num_tokens),
            "pixel_values": torch.full((2, 3), 2.0),
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
            "mm_modalities": [Modality.IMAGE, Modality.IMAGE],
            "mm_spans": [[(1, 3)], [(3, 5)]],
            "mrope_positions": torch.arange(3 * num_tokens).view(3, num_tokens),
            "mrope_position_delta": torch.tensor([[0]]),
        }


def _rebuild(monkeypatch, multimodal_inputs, tokens):
    processor = _RecordingProcessor()
    monkeypatch.setattr(
        token_space_media_process, "_token_space_processor", lambda hf_checkpoint, mm_process_config_json: processor
    )
    train_inputs = token_space_media_process.compute_multimodal_train_inputs(
        "ckpt", None, multimodal_inputs, tokens, device=torch.device("cpu")
    )
    return processor, train_inputs


def test_trainer_rebuilds_tensors_without_expanding_the_rollout_tokens(monkeypatch):
    processor, train_inputs = _rebuild(
        monkeypatch, [{"images": ["/shared/a.png", "/shared/b.png"]}], [torch.tensor([1, 9, 9, 9, 9, 2, 5])]
    )

    (call,) = processor.calls
    # The rollout tokens are already expanded, so expansion starts past their end.
    assert call["input_ids"] == [1, 9, 9, 9, 9, 2, 5]
    assert call["mm_token_expansion_start_len"] == 7
    assert call["image_data"] == ["/shared/a.png", "/shared/b.png"]
    assert call["image_device"] == "cpu"
    assert torch.equal(train_inputs[0]["image_grid_thw"], torch.tensor([[1, 2, 2], [1, 2, 2]]))


def test_placement_and_mrope_come_from_sglang(monkeypatch):
    _, (media, text) = _rebuild(
        monkeypatch,
        [{"images": ["/shared/a.png", "/shared/b.png"]}, None],
        [torch.tensor([1, 9, 9, 9, 9, 2, 5]), torch.tensor([1, 2])],
    )

    assert set(media) == {"pixel_values", "image_grid_thw", "mm_image_positions", "mrope_position_ids"}
    # Every tower output row, at the tokens of its media's half-open spans.
    assert media["mm_image_positions"].tolist() == [1, 2, 3, 4]
    # Token-major, so samples concatenate along dim 0.
    assert media["mrope_position_ids"].shape == (7, 3)
    assert media["mrope_position_ids"][:, 1].tolist() == list(range(7, 14))
    assert text["mrope_position_ids"].tolist() == [[0, 0, 0], [1, 1, 1]]
