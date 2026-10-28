# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The MiniMax-H3 requests: text, or text with first/last keyframes (FL2VA), or
text with reference images, videos and audio (Ref2VA), to a video with its
soundtrack. A deployment serves FL2VA or Ref2VA, which load different
transformers, so each is its own model.

The served envelope mirrors tt-metal's models/tt_dit/pipelines/minimax_h3
(packing.py and policy.py), which also rejects anything outside it.
"""

from typing import Annotated, Literal, Optional

from domain.generation_request import (
    AudioMedia,
    GenerationRequest,
    Media,
    Seed,
    VideoMedia,
)
from pydantic import BeforeValidator, Field, field_validator, model_validator



def _normalized_ratio(value):
    """``16x9``, ``16/9`` and `` 16 : 9 `` as ``16:9``, as the earlier H3 API
    took them."""
    if not isinstance(value, str):
        return value
    parts = value.strip().replace("x", ":").replace("/", ":").split(":")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        return value
    return f"{int(parts[0])}:{int(parts[1])}"


H3Ratio = Annotated[
    Literal["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
    BeforeValidator(_normalized_ratio),
]
H3_FPS = 24
H3_SHORT_EDGE = 768
H3_MAX_PIXELS = 768 * 1344
H3_CANVAS_MULTIPLE = 32
# The VAE encodes 17n + 5 frames.
H3_FRAMES_PER_CHUNK = 17
H3_LATENTS_PER_CHUNK = 5
H3_KEYFRAME_MIN_SIDE = 256
H3_KEYFRAME_MAX_SIDE = 5760
# Ref2VA reference clips: the product's limits; the pipeline truncates instead.
H3_CLIP_MIN_S = 2.0
H3_CLIP_MAX_S = 15.0
H3_CLIPS_MAX_S = 15.0


def h3_num_frames(duration: int) -> int:
    """The frames generated for ``duration`` seconds: the 24 fps count, rounded
    up to the next 17n + 5."""
    frames = round(duration * H3_FPS)
    while frames % H3_FRAMES_PER_CHUNK != H3_LATENTS_PER_CHUNK:
        frames += 1
    return frames


def h3_canvas(width: float, height: float) -> tuple[int, int]:
    """(width, height) of the output for a shape: the short side scaled to 768,
    both scaled down if the area is then above 768 × 1344, each rounded to a
    multiple of 32."""
    ratio = width / height
    if ratio >= 1:
        width, height = H3_SHORT_EDGE * ratio, float(H3_SHORT_EDGE)
    else:
        width, height = float(H3_SHORT_EDGE), H3_SHORT_EDGE / ratio
    if width * height > H3_MAX_PIXELS:
        scale = (H3_MAX_PIXELS / (width * height)) ** 0.5
        width, height = width * scale, height * scale
    multiple = H3_CANVAS_MULTIPLE
    return (
        max(multiple, round(width / multiple) * multiple),
        max(multiple, round(height / multiple) * multiple),
    )


def _within_1_4_and_4_1(width: int, height: int) -> bool:
    return width <= 4 * height and height <= 4 * width


class _MiniMaxH3Request(GenerationRequest):
    OUTPUT_TYPE = "video"
    AUDIO = True
    INLINE_MEDIA_MAX_CHARS = 64_000_000
    # Reference images can each decode to hundreds of MB, too much to send to
    # every rank of a multi-host mesh.
    RUNNER_DECODES_IMAGES = True

    prompt: str = Field(
        min_length=1,
        description="Describes the picture and the sound. At most 3008 tokens.",
    )
    aspect_ratio: H3Ratio = Field(
        default="16:9",
        description=(
            "Output shape, unless `width` and `height` are set. The size follows: "
            "the short side 768, the area at most 768 × 1344, each side a "
            "multiple of 32 (`16:9` 1344×768, `21:9` 1536×672)."
        ),
    )
    width: Optional[int] = Field(
        default=None,
        ge=H3_CANVAS_MULTIPLE,
        multiple_of=H3_CANVAS_MULTIPLE,
        description=(
            "With `height`, the output shape in place of `aspect_ratio`: a "
            "multiple of 32, `width × height` at most 1,032,192, the ratio from "
            "1:4 to 4:1. It is resized like `aspect_ratio`, so it sets the shape, "
            "not the size: 832 × 480 gives 1344×768."
        ),
    )
    height: Optional[int] = Field(
        default=None,
        ge=H3_CANVAS_MULTIPLE,
        multiple_of=H3_CANVAS_MULTIPLE,
        description="See `width`.",
    )
    duration: int = Field(default=5, ge=4, le=15, description="Seconds.")
    seed: Optional[Seed] = None

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return value

    @model_validator(mode="after")
    def _canvas(self):
        if (self.width is None) != (self.height is None):
            raise ValueError("set both width and height, or neither")
        if self.width is not None:
            if self.width * self.height > H3_MAX_PIXELS:
                raise ValueError(
                    f"width × height must be at most {H3_MAX_PIXELS}; got "
                    f"{self.width} × {self.height}"
                )
            if not _within_1_4_and_4_1(self.width, self.height):
                raise ValueError("width : height must be from 1:4 to 4:1")
        return self

    @property
    def fps(self) -> int:
        return H3_FPS

    @property
    def frames(self) -> int:
        return h3_num_frames(self.duration)

    def output_size(self) -> tuple[int, int]:
        if self.width is not None:
            return h3_canvas(self.width, self.height)
        return h3_canvas(*map(int, self.aspect_ratio.split(":")))

    @classmethod
    def warmup_sizes(cls) -> list[tuple[int, int]]:
        # The pipeline warms itself up.
        return []


class _MiniMaxH3KeyframesRequest(_MiniMaxH3Request):
    MEDIA_FIELDS = ("first_frame", "last_frame")

    first_frame: Optional[Media] = Field(
        default=None,
        description=(
            "The image the video starts from. Each side 256 to 5760 px, the ratio "
            "from 1:4 to 4:1. Stretched to the output size. Without `width` and "
            "`height`, the output takes its shape instead of `aspect_ratio`'s."
        ),
    )
    last_frame: Optional[Media] = Field(
        default=None,
        description=(
            "The image the video ends on, with the same limits. With a "
            "`first_frame`, resized to fill the output and center-cropped; alone, "
            "stretched, and the output takes its shape as for `first_frame`."
        ),
    )

    def media_errors(self) -> list[tuple[str, int, str]]:
        errors = []
        for name, images in self._images.items():
            width, height = images[0].size
            if not (
                H3_KEYFRAME_MIN_SIDE <= min(width, height)
                and max(width, height) <= H3_KEYFRAME_MAX_SIDE
            ):
                errors.append(
                    (
                        name,
                        0,
                        f"is {width}×{height}; each side must be from "
                        f"{H3_KEYFRAME_MIN_SIDE} to {H3_KEYFRAME_MAX_SIDE} px",
                    )
                )
            elif not _within_1_4_and_4_1(width, height):
                errors.append(
                    (name, 0, f"is {width}×{height}; the ratio must be from 1:4 to 4:1")
                )
        return errors

    def output_size(self) -> tuple[int, int]:
        keyframe = self._images.get("first_frame") or self._images.get("last_frame")
        if self.width is None and keyframe:
            return h3_canvas(*keyframe[0].size)
        return super().output_size()


class MiniMaxH3Request(_MiniMaxH3KeyframesRequest):
    MODEL_ID = "minimaxai/minimax-h3"


class MiniMaxH3TurboRequest(_MiniMaxH3KeyframesRequest):
    MODEL_ID = "lightx2v/minimax-h3-turbo"


class MiniMaxH3FastH3Request(_MiniMaxH3Request):
    MODEL_ID = "minimaxai/minimax-h3-fasth3"


class _MiniMaxH3ReferencesRequest(_MiniMaxH3Request):
    MEDIA_FIELDS = ("images", "videos", "audios")
    MEDIA_KINDS = {"videos": "video", "audios": "audio"}

    images: list[Media] = Field(
        default=[],
        max_length=9,
        description="Reference images, each with a ratio from 1:4 to 4:1.",
    )
    videos: list[VideoMedia] = Field(
        default=[],
        max_length=3,
        description=(
            "Reference videos, each 2 to 15 s with a ratio from 1:4 to 4:1, at "
            "most 15 s together. A video's soundtrack is a reference too."
        ),
    )
    audios: list[AudioMedia] = Field(
        default=[],
        max_length=3,
        description=(
            "Reference audio clips, each 2 to 15 s, at most 15 s together. Only "
            "with at least one image or video."
        ),
    )

    @model_validator(mode="after")
    def _references(self):
        if not (self.images or self.videos):
            raise ValueError("needs at least one reference image or video")
        if len(self.images) + len(self.videos) + len(self.audios) > 12:
            raise ValueError("at most 12 references in total")
        return self

    def media_errors(self) -> list[tuple[str, int, str]]:
        errors = []
        for index, image in enumerate(self._images.get("images", [])):
            if not _within_1_4_and_4_1(*image.size):
                errors.append(
                    ("images", index, f"is {image.size[0]}×{image.size[1]}; the "
                     "ratio must be from 1:4 to 4:1")
                )
        for name in ("videos", "audios"):
            probes = self._probes.get(name, [])
            for index, probe in enumerate(probes):
                if not H3_CLIP_MIN_S <= probe.duration_s <= H3_CLIP_MAX_S:
                    errors.append(
                        (name, index, f"is {probe.duration_s:g} s; each clip must "
                         f"be {H3_CLIP_MIN_S:g} to {H3_CLIP_MAX_S:g} s")
                    )
                if name == "videos":
                    if probe.size is None:
                        errors.append((name, index, "has no video stream"))
                    elif not _within_1_4_and_4_1(*probe.size):
                        errors.append(
                            (name, index, f"is {probe.size[0]}×{probe.size[1]}; "
                             "the ratio must be from 1:4 to 4:1")
                        )
                    if not probe.fps:
                        errors.append((name, index, "reports no frame rate"))
                elif probe.audio_channels is None:
                    errors.append((name, index, "has no audio stream"))
                if probe.audio_channels not in (None, 1, 2):
                    errors.append(
                        (name, index, f"has {probe.audio_channels} audio channels; "
                         "must be mono or stereo")
                    )
            total = sum(probe.duration_s for probe in probes)
            if total > H3_CLIPS_MAX_S:
                errors.append(
                    (name, len(probes) - 1, f"the clips are {total:g} s together; "
                     f"at most {H3_CLIPS_MAX_S:g} s")
                )
        return errors


class MiniMaxH3Ref2VARequest(_MiniMaxH3ReferencesRequest):
    MODEL_ID = "minimaxai/minimax-h3-ref2va"


class MiniMaxH3TurboRef2VARequest(_MiniMaxH3ReferencesRequest):
    MODEL_ID = "lightx2v/minimax-h3-turbo-ref2va"
