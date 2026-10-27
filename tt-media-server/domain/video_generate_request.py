# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os
from typing import Optional

from config.constants import (
    DEFAULT_VIDEO_INFERENCE_STEPS,
    MAX_VIDEO_INFERENCE_STEPS,
    MIN_VIDEO_INFERENCE_STEPS,
)
from config.settings import get_settings
from domain.base_request import BaseRequest
from pydantic import Field, field_validator, model_validator

# Fields on the shared schema that MiniMax-H3 does not read: the frame-shape
# fields, ``num_inference_steps`` because H3's AdaLN table is precomputed at a
# fixed step count, and ``negative_prompt`` because the H3 pipeline takes none.
_H3_UNREAD_FIELDS = frozenset(
    {"fps", "num_frames", "num_inference_steps", "negative_prompt"}
)


class VideoGenerateRequest(BaseRequest):
    # Required fields
    prompt: str

    @field_validator("prompt")
    @classmethod
    def _prompt_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return value

    # Optional fields
    negative_prompt: Optional[str] = None
    # None = the default; resolved in _default_steps.
    num_inference_steps: Optional[int] = Field(
        default=None,
        ge=MIN_VIDEO_INFERENCE_STEPS,
        le=MAX_VIDEO_INFERENCE_STEPS,
    )
    seed: Optional[int] = None

    # TODO: Make generic for all video models, and remove model specific logic
    # Output shape. Both are model-specific and both are optional: a model that serves one fixed
    # shape ignores them, and MiniMax-H3 t2va resolves them against its published working points
    # (see `minimax_h3_parse_aspect_ratio` / `MINIMAX_H3_DURATIONS_S`). Left as free-form here
    # rather than an enum so MiniMax-H3's admission at submit can reject with a message naming what
    # it does serve -- a 422 from pydantic on a shared field cannot say that.
    aspect_ratio: Optional[str] = Field(default=None, examples=["16:9", "9:16", "1:1"])

    # TODO: Make generic for all video models, and remove model specific logic
    # Unknown fields are refused for MiniMax-H3 rather than ignored. Pydantic's default is to
    # drop them silently, which meant `{"resolution": "1080P", "model": "NotMiniMax", "duration": 9}`
    # came back 202 with none of it applied -- the caller believes it asked for something it did
    # not get, which is worse than an error. Scoped to H3 by runner rather than set as
    # `extra="forbid"` on the class, because Wan clients share this model and may send fields this
    # deployment does not read.
    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_fields(cls, data):
        if not isinstance(data, dict) or not _is_minimax_h3():
            return data
        readable = set(cls.model_fields) - _H3_UNREAD_FIELDS
        # A null for a field H3 does not read asks for nothing, so it is not refused. Form
        # endpoints pass these as None when the client leaves them out.
        supplied = {
            key
            for key, value in data.items()
            if value is not None or key not in _H3_UNREAD_FIELDS
        }
        unknown = sorted(supplied - readable)
        if unknown:
            known = ", ".join(sorted(readable))
            raise ValueError(
                f"unknown field(s) for MiniMax-H3: {', '.join(unknown)}. "
                f"This deployment reads: {known}. Resolution is selected with `aspect_ratio` "
                "or `height`/`width`."
            )
        return data

    # Shape. None = the model's default.
    # MiniMax-H3: an explicit height/width canvas overrides `aspect_ratio`. Left unset, the
    # pipeline resolves it from `aspect_ratio`, or from the first keyframe for fl2va.
    height: Optional[int] = Field(default=None, gt=0)
    width: Optional[int] = Field(default=None, gt=0)
    fps: Optional[float] = Field(default=None, gt=0)
    duration: Optional[float] = Field(default=None, gt=0)
    num_frames: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _default_steps(self):
        if self.num_inference_steps is None:
            self.num_inference_steps = DEFAULT_VIDEO_INFERENCE_STEPS
        return self


# TODO: Remove model specific logic
def _is_minimax_h3() -> bool:
    from config.constants import ModelNames, ModelRunners

    try:
        runner = get_settings().model_runner
    except Exception:  # noqa: BLE001 - settings unavailable (tests, tooling): do not gate on it
        return False

    if runner in {
        ModelRunners.TT_MINIMAX_H3_T2VA.value,
        ModelRunners.TT_MINIMAX_H3_FL2VA.value,
        ModelRunners.TT_MINIMAX_H3_REF2VA.value,
        ModelRunners.TT_MINIMAX_H3_FASTH3.value,
        ModelRunners.TT_MINIMAX_H3_FL2VA_LIGHTX2V.value,
        ModelRunners.TT_MINIMAX_H3_REF2VA_LIGHTX2V.value,
    }:
        return True

    # sp_runner is a SHM proxy and does not load weights. MODEL is the same
    # peer signal the video API already uses for T2VA / FL2VA / Ref2VA routing;
    # without it a 30s request would 202 and then fail as a worker error.
    if runner != ModelRunners.SP_RUNNER.value:
        return False
    model_env = os.getenv("MODEL")
    if not model_env:
        return False
    try:
        return ModelNames(model_env) in {
            ModelNames.MINIMAX_H3,
            ModelNames.MINIMAX_H3_FL2VA,
            ModelNames.MINIMAX_H3_REF2VA,
            ModelNames.MINIMAX_H3_FASTH3,
            ModelNames.MINIMAX_H3_FL2VA_LIGHTX2V,
            ModelNames.MINIMAX_H3_REF2VA_LIGHTX2V,
        }
    except ValueError:
        return False
