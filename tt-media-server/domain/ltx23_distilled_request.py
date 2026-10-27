# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from typing import Optional

from config.constants import (
    LTX_FPS,
    LTX_HEIGHT,
    LTX_NUM_FRAMES,
    LTX_WIDTH,
    snap_num_frames,
)
from domain.generation_request import GenerationRequest, Seed
from pydantic import Field, model_validator

_SHAPE = (
    f"{LTX_WIDTH}×{LTX_HEIGHT}, {LTX_NUM_FRAMES} frames at {LTX_FPS:g} fps "
    f"({LTX_NUM_FRAMES / LTX_FPS:.2f} s)"
)


class Ltx23DistilledRequest(GenerationRequest):
    MODEL_ID = "lightricks/ltx-2.3-distilled"
    OUTPUT_TYPE = "video"
    AUDIO = True

    prompt: str = Field(min_length=1)
    width: int = Field(
        default=LTX_WIDTH,
        ge=LTX_WIDTH,
        le=LTX_WIDTH,
        description=f"Only {LTX_WIDTH}: the pipeline is built and traced for {_SHAPE}.",
    )
    height: int = Field(
        default=LTX_HEIGHT,
        ge=LTX_HEIGHT,
        le=LTX_HEIGHT,
        description=f"Only {LTX_HEIGHT}: the pipeline is built and traced for {_SHAPE}.",
    )
    frames: int = Field(
        default=LTX_NUM_FRAMES,
        ge=LTX_NUM_FRAMES,
        le=LTX_NUM_FRAMES,
        description=f"Only {LTX_NUM_FRAMES}: the pipeline is built and traced for {_SHAPE}.",
    )
    duration: Optional[float] = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Seconds, an alternative to `frames`: converted at `fps` and snapped "
            f"to the nearest frame count of the form 8k + 1, which must be "
            f"{LTX_NUM_FRAMES}. Given with `frames`, the two must agree."
        ),
    )
    fps: int = Field(
        default=int(LTX_FPS),
        ge=int(LTX_FPS),
        le=int(LTX_FPS),
        description=(
            f"Only {LTX_FPS:g}: the rate sets the audio length and the audio-video "
            "positions, which are baked into the traces."
        ),
    )
    seed: Optional[Seed] = None

    @model_validator(mode="after")
    def _duration_gives_the_frames(self):
        if self.duration is None:
            return self
        frames = snap_num_frames(round(self.duration * self.fps))
        if frames != self.frames:
            raise ValueError(
                f"duration={self.duration:g} s is {frames} frames at {self.fps} fps; "
                f"this deployment serves {_SHAPE}"
            )
        return self

    def output_size(self) -> tuple[int, int]:
        return self.width, self.height

    @classmethod
    def warmup_sizes(cls) -> list[tuple[int, int]]:
        return [(LTX_WIDTH, LTX_HEIGHT)]
