# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import re
from fractions import Fraction
from typing import Annotated, ClassVar, Literal

from domain.base_request import BaseRequest
from PIL import Image
from pydantic import AfterValidator, ConfigDict, Field, PrivateAttr

MAX_SEED = 2**31 - 1

MediaKind = Literal["image", "video", "audio"]
# Per file, inline or fetched.
MEDIA_MAX_BYTES: dict[str, int] = {
    "image": 30_000_000,
    "video": 50_000_000,
    "audio": 15_000_000,
}


def _media_checker(kind: MediaKind, example: str):
    data_uri = re.compile(rf"^data:{kind}/[a-z0-9.+-]+;base64,", re.IGNORECASE)

    def check(value: str) -> str:
        if value.startswith("https://") or data_uri.match(value):
            return value
        raise ValueError(f"must be an https:// URL or a base64 data URI ({example})")

    return AfterValidator(check)


Media = Annotated[
    str,
    _media_checker("image", "data:image/png;base64,..."),
    Field(
        description=(
            "An `https://` URL, which the server fetches when the job is "
            "submitted, or a data URI (`data:image/png;base64,…`). PNG, JPEG or "
            "WebP, at most 30 MB per image."
        )
    ),
]
VideoMedia = Annotated[
    str,
    _media_checker("video", "data:video/mp4;base64,..."),
    Field(
        description=(
            "An `https://` URL, which the server fetches when the job is "
            "submitted, or a data URI (`data:video/mp4;base64,…`). MP4 or MOV, "
            "H.264 or H.265 video, optionally with an AAC or MP3 audio track; at "
            "most 50 MB."
        )
    ),
]
AudioMedia = Annotated[
    str,
    _media_checker("audio", "data:audio/wav;base64,..."),
    Field(
        description=(
            "An `https://` URL, which the server fetches when the job is "
            "submitted, or a data URI (`data:audio/wav;base64,…`). WAV or MP3, "
            "mono or stereo; at most 15 MB."
        )
    ),
]
Seed = Annotated[
    int,
    Field(
        ge=0,
        le=MAX_SEED,
        description="Omitted = random. Each output reports the seed it was made with.",
    ),
]
ImageFormat = Literal["png", "jpeg", "webp"]
Count = Annotated[
    int,
    Field(
        ge=1,
        le=1,
        description=(
            "Number of images. Only 1 for now; kept so more can be allowed later "
            "without a schema change."
        ),
    ),
]


def closest_ratio(width: int, height: int, ratios) -> str:
    """The ratio in ``ratios`` ("W:H" strings) closest to ``width`` x
    ``height`` by factor. The distances are exact, so a tie, such as 4:3
    between 16:9 and 1:1, goes to the earlier ratio in ``ratios``."""
    target = Fraction(width, height)

    def distance(ratio: str) -> Fraction:
        value = Fraction(*map(int, ratio.split(":")))
        return max(value / target, target / value)

    return min(ratios, key=distance)


# The request of one model's submit endpoint. The service fills in the private
# attributes before queueing it; the model's runner then consumes it directly.
class GenerationRequest(BaseRequest):
    model_config = ConfigDict(extra="forbid")

    # The `{org}/{model}` of the submit path, the upstream model's lowercase
    # Hugging Face ID.
    MODEL_ID: ClassVar[str]
    OUTPUT_TYPE: ClassVar[Literal["image", "video"]]
    # Fields holding media, alone or in a list.
    MEDIA_FIELDS: ClassVar[tuple[str, ...]] = ()
    # The kind of each media field that does not hold images.
    MEDIA_KINDS: ClassVar[dict[str, MediaKind]] = {}
    # Whether the runner decodes the input images from their files. The
    # request then carries files rather than pixels to the device worker and
    # to the other ranks of a multi-host mesh: a decoded image can be a
    # hundred times its file.
    RUNNER_DECODES_IMAGES: ClassVar[bool] = False

    _images: dict[str, list[Image.Image]] = PrivateAttr(default_factory=dict)
    # The input files, and the metadata probed from videos and audio clips.
    _files: dict[str, list[bytes]] = PrivateAttr(default_factory=dict)
    _probes: dict[str, list[object]] = PrivateAttr(default_factory=dict)
    _width: int = PrivateAttr(default=0)
    _height: int = PrivateAttr(default=0)
    _seed: int = PrivateAttr(default=0)
    _prompt: str = PrivateAttr(default="")
    # Shared with the device worker: a set cancel event keeps a worker from
    # claiming the request, and the assignment tells the job manager whether
    # one already has.
    _cancel_event: object = PrivateAttr(default=None)
    _worker_assignment: object = PrivateAttr(default=None)

    def media(self) -> dict[str, list[str]]:
        media = {}
        for name in self.MEDIA_FIELDS:
            value = getattr(self, name)
            if value is not None:
                media[name] = value if isinstance(value, list) else [value]
        return media

    @classmethod
    def media_kind(cls, name: str) -> MediaKind:
        return cls.MEDIA_KINDS.get(name, "image")

    def media_errors(self) -> list[tuple[str, int, str]]:
        """(field, index, message) for each loaded input the model cannot
        take, from what the schema cannot see: image sizes, clip lengths."""
        return []

    def output_size(self) -> tuple[int, int]:
        """(width, height) of the outputs, from the fields and the decoded input
        images."""
        raise NotImplementedError

    @classmethod
    def warmup_sizes(cls) -> list[tuple[int, int]]:
        """Every (width, height) a request can produce, as far as they can be
        listed, so the runner can compile and trace them before serving."""
        raise NotImplementedError

    def model_prompt(self) -> str:
        """The prompt as the model gets it."""
        return getattr(self, "prompt")
