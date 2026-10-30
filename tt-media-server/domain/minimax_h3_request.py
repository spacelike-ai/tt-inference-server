# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The MiniMax-H3 requests: text, optionally with first/last keyframes (FL2VA),
or text with reference images, videos and audio (Ref2VA), to a video with its
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
# `adaptive` follows the keyframe's shape.
H3KeyframesRatio = Annotated[
    Literal["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
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
# The audio clips plus the reference videos with a soundtrack.
H3_AUDIO_REFERENCES_MAX = 3


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
    # Reference images can each decode to hundreds of MB, too much to send to
    # every rank of a multi-host mesh.
    RUNNER_DECODES_IMAGES = True

    prompt: str = Field(min_length=1)
    aspect_ratio: H3Ratio = Field(
        default="16:9",
        description="Output shape. Each output reports its exact size.",
    )
    size: Literal["768p"] = Field(
        default="768p",
        description=(
            "Size tier: a short edge of 768, the area capped at 768×1344, sides "
            "multiples of 32 (`16:9` 1344×768, `21:9` 1536×672). Only `768p` for "
            "now; kept so other tiers can be added without a schema change."
        ),
    )
    duration: int = Field(
        default=5,
        ge=4,
        le=15,
        description=(
            "Seconds. The video runs at 24 fps and its length is rounded up to the "
            "frame counts the model generates (17k + 5): 5 s gives 124 frames "
            "(5.17 s). Each output reports its exact `duration` and `num_frames`."
        ),
    )
    seed: Optional[Seed] = None

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return value

    @property
    def fps(self) -> int:
        return H3_FPS

    @property
    def frames(self) -> int:
        return h3_num_frames(self.duration)

    def output_size(self) -> tuple[int, int]:
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
            "The frame the video starts on. 256–5760 pixels per side, aspect "
            "ratio from 1:4 to 4:1."
        ),
    )
    last_frame: Optional[Media] = Field(
        default=None,
        description=(
            "The frame the video ends on. Allowed with or without `first_frame`. "
            "Same limits as `first_frame`."
        ),
    )
    aspect_ratio: Optional[H3KeyframesRatio] = Field(
        default=None,
        description=(
            "Output shape. With a frame: only `adaptive` (the default), meaning "
            "the output follows the first frame's shape, or the last frame's when "
            "there is no first frame. Without frames: one of the listed ratios, "
            "default `16:9`. Each output reports its exact size."
        ),
    )

    @model_validator(mode="after")
    def _adaptive_with_frames(self):
        has_frame = self.first_frame is not None or self.last_frame is not None
        if self.aspect_ratio is None:
            self.aspect_ratio = "adaptive" if has_frame else "16:9"
        elif has_frame and self.aspect_ratio != "adaptive":
            raise ValueError(
                "with a first_frame or last_frame, aspect_ratio must be adaptive: "
                "the output follows the frame's shape"
            )
        elif not has_frame and self.aspect_ratio == "adaptive":
            raise ValueError("aspect_ratio adaptive needs a first_frame or last_frame")
        return self

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
        if self.aspect_ratio == "adaptive":
            keyframe = self._images.get("first_frame") or self._images["last_frame"]
            return h3_canvas(*keyframe[0].size)
        return super().output_size()


class MiniMaxH3FL2VARequest(_MiniMaxH3KeyframesRequest):
    MODEL_ID = "minimaxai/minimax-h3-fl2va"


class MiniMaxH3FL2VALightX2VRequest(_MiniMaxH3KeyframesRequest):
    MODEL_ID = "tenstorrent/minimax-h3-fl2va-lightx2v"


class _MiniMaxH3ReferencesRequest(_MiniMaxH3Request):
    """References: at most 12 in total, and at least one image or video. Video
    and audio clips are 2–15 s each and at most 15 s combined per kind. A
    reference video's soundtrack counts toward the audio limits as well as the
    video ones, and once toward the 12-reference total."""

    MEDIA_FIELDS = ("images", "videos", "audios")
    MEDIA_KINDS = {"videos": "video", "audios": "audio"}

    prompt: str = Field(
        min_length=1,
        description=(
            "Refer to references by kind and position, numbered per kind in the "
            "order given: `Picture 1` (the first of `images`), `Video 1`, `Audio "
            "1`. The model sees each reference under that label. A video's own "
            "sound track counts as an audio reference too."
        ),
    )
    images: list[Media] = Field(
        default=[],
        max_length=9,
        description="Reference images, aspect ratio from 1:4 to 4:1.",
    )
    videos: list[VideoMedia] = Field(
        default=[],
        max_length=3,
        description=(
            "Reference video clips, 2–15 s each, at most 15 s combined. A clip's "
            "soundtrack, if it has one, is also audio conditioning and counts "
            "toward the audio limits (see `audios`)."
        ),
    )
    audios: list[AudioMedia] = Field(
        default=[],
        max_length=3,
        description=(
            "Reference audio clips, 2–15 s each. The audio limits cover these "
            "clips plus the soundtrack of every reference video that has one: at "
            "most 3 audio references and at most 15 s of audio combined."
        ),
    )
    aspect_ratio: H3Ratio = Field(
        default="16:9",
        description=(
            "Output shape, as for text-to-video: the references don't set it. "
            "Each output reports its exact size."
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
        videos = self._probes.get("videos", [])
        total = sum(probe.duration_s for probe in videos)
        if total > H3_CLIPS_MAX_S:
            errors.append(
                ("videos", len(videos) - 1, f"the clips are {total:g} s together; "
                 f"at most {H3_CLIPS_MAX_S:g} s")
            )
        # The soundtracks of the videos alone stay within the audio limits, or
        # the videos are over theirs; going over takes an audio clip, and the
        # error points at the last one.
        audios = self._probes.get("audios", [])
        if audios:
            sounds = audios + [probe for probe in videos if probe.audio_channels]
            if len(sounds) > H3_AUDIO_REFERENCES_MAX:
                errors.append(
                    ("audios", len(audios) - 1, f"with the videos' soundtracks, "
                     f"{len(sounds)} audio references; at most "
                     f"{H3_AUDIO_REFERENCES_MAX}")
                )
            total = sum(probe.duration_s for probe in sounds)
            if total > H3_CLIPS_MAX_S:
                errors.append(
                    ("audios", len(audios) - 1, f"with the videos' soundtracks, "
                     f"{total:g} s of audio together; at most {H3_CLIPS_MAX_S:g} s")
                )
        return errors


class MiniMaxH3Ref2VARequest(_MiniMaxH3ReferencesRequest):
    MODEL_ID = "minimaxai/minimax-h3-ref2va"


class MiniMaxH3Ref2VALightX2VRequest(_MiniMaxH3ReferencesRequest):
    MODEL_ID = "tenstorrent/minimax-h3-ref2va-lightx2v"
