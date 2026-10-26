"""Multimodal training inputs computed on the trainer from shared media references.

With ``--multimodal-token-space-processing`` the rollout ships ordered media references instead of
processor tensors. Each trainer rank rebuilds the tensors with SGLang's ``TokenSpaceProcessor``,
the same core the engines serve with, built from the engines' ``--sglang-mm-process-config`` and
run on the device the engine uses, so training sees the features the rollout engine saw.

Besides the processor tensors, each sample carries what SGLang computes from the expanded tokens:

- ``mm_image_positions`` / ``mm_video_positions``: [num_rows] sample-local token index of every vision tower
  output row, from the processor's ``mm_spans``;
- ``mrope_position_ids``: [num_tokens, 3] mRoPE positions, token-major so the samples of a micro-batch
  concatenate along dim 0. Text-only samples carry them too.
"""

import asyncio
import functools
import json
from collections.abc import Sequence

import torch

from miles.utils.processing_utils import extract_multimodal_train_inputs


def compute_multimodal_train_inputs(
    hf_checkpoint: str,
    mm_process_config: dict | None,
    multimodal_inputs: Sequence[dict[str, list[str]] | None],
    tokens: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> list[dict[str, torch.Tensor] | None]:
    """Rebuild each sample's processor tensors on the trainer from its media references and expanded tokens.

    Args:
        hf_checkpoint: The checkpoint whose config and processor SGLang serves.
        mm_process_config: The engines' ``--sglang-mm-process-config``.
        multimodal_inputs: Each sample's media references, or None for a text-only sample.
        tokens: Each sample's tokens, with the media placeholders SGLang expanded.
        device: Where the image and video processors run, as on the engine, and where the tensors land.

    Returns:
        Each sample's ``multimodal_train_inputs``; a text-only sample has only its mRoPE positions, or None for a
        model without mRoPE.
    """
    # Canonical JSON so the config can key the processor cache.
    processor = _token_space_processor(hf_checkpoint, json.dumps(mm_process_config or {}, sort_keys=True))
    return [
        (
            _sample_train_inputs(processor, sample_multimodal_inputs, sample_tokens, device=device)
            if sample_multimodal_inputs
            else _text_train_inputs(processor, sample_tokens)
        )
        for sample_multimodal_inputs, sample_tokens in zip(multimodal_inputs, tokens, strict=True)
    ]


@functools.cache
def _token_space_processor(hf_checkpoint: str, mm_process_config_json: str):
    # Deferred: only SGLang builds with the token-space processor provide these modules.
    from sglang.srt.multimodal.processors.processor_config import MultimodalProcessorConfig
    from sglang.srt.multimodal.token_space_mm_process.processor import TokenSpaceProcessor
    from sglang.srt.multimodal.token_space_mm_process.registry import strategy_for
    from sglang.srt.utils.hf_transformers import get_config, get_processor

    hf_config = get_config(hf_checkpoint, trust_remote_code=True)
    strategy_class = strategy_for(hf_config)
    if strategy_class is None:
        raise ValueError(f"--multimodal-token-space-processing does not support {hf_config.architectures}")
    # The engine's default image processor backend; the engine resolves "auto" the same way.
    hf_processor = get_processor(hf_checkpoint, trust_remote_code=True, image_processor_backend="auto")
    return TokenSpaceProcessor.for_strategy(strategy_class)(
        hf_config,
        hf_processor,
        processor_config=MultimodalProcessorConfig(
            image_processor_backend="auto", mm_process_config=json.loads(mm_process_config_json)
        ),
        compute_mrope_positions=True,
    )


def _sample_train_inputs(
    processor, multimodal_inputs: dict[str, list[str]], tokens: torch.Tensor, *, device: torch.device
):
    from sglang.srt.multimodal.modality import Modality

    input_ids = tokens.tolist()
    batch_feature = asyncio.run(
        processor.process(
            image_data=multimodal_inputs.get("images"),
            video_data=multimodal_inputs.get("videos"),
            audio_data=multimodal_inputs.get("audios"),
            input_ids=input_ids,
            mm_token_expansion_start_len=len(input_ids),
            image_device=str(device),
            video_device=str(device),
        )
    )
    train_inputs = extract_multimodal_train_inputs(batch_feature)
    modalities, spans = train_inputs.pop("mm_modalities"), train_inputs.pop("mm_spans")
    mrope_positions = train_inputs.pop("mrope_positions", None)
    train_inputs.pop("mrope_position_delta", None)
    train_inputs = {key: value.to(device) for key, value in train_inputs.items()}
    # The token index of every tower output row, from where SGLang placed each media (half-open spans).
    for modality, key in ((Modality.IMAGE, "mm_image_positions"), (Modality.VIDEO, "mm_video_positions")):
        rows = [
            torch.arange(start, end)
            for media, media_spans in zip(modalities, spans, strict=True)
            if media == modality
            for start, end in media_spans
        ]
        if rows:
            train_inputs[key] = torch.cat(rows).to(device)
    if mrope_positions is not None:
        train_inputs["mrope_position_ids"] = mrope_positions.T.contiguous().to(device)
    return train_inputs


def _text_train_inputs(processor, tokens: torch.Tensor):
    mrope_positions, _ = processor.strategy.compute_mrope_positions(tokens.tolist(), {})
    if mrope_positions is None:
        return None
    return {"mrope_position_ids": mrope_positions.T.contiguous().to(tokens.device)}
