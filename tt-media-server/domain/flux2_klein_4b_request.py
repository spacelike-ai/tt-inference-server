# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from typing import Literal, Optional

from domain.generation_request import (
    Count,
    GenerationRequest,
    ImageFormat,
    Media,
    Seed,
    closest_ratio,
)
from pydantic import Field

KleinRatio = Literal["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
KleinSize = Literal["1K", "2K"]

# (width, height) per size tier and ratio: about 1 MP at 1K, twice the sides at
# 2K, multiples of 16 as the VAE and patch size need.
_KLEIN_1K_SIZES = {
    "1:1": (1024, 1024),
    "16:9": (1376, 768),
    "9:16": (768, 1376),
    "4:3": (1184, 896),
    "3:4": (896, 1184),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
}
KLEIN_SIZES: dict[str, dict[str, tuple[int, int]]] = {
    "1K": _KLEIN_1K_SIZES,
    "2K": {ratio: (2 * w, 2 * h) for ratio, (w, h) in _KLEIN_1K_SIZES.items()},
}
KLEIN_STEPS = 4


class Flux2klein4bRequest(GenerationRequest):
    MODEL_ID = "black-forest-labs/flux.2-klein-4b"
    OUTPUT_TYPE = "image"
    MEDIA_FIELDS = ("images",)

    prompt: str = Field(min_length=1)
    images: Optional[list[Media]] = Field(
        default=None,
        max_length=4,
        description="Images to edit or draw from. Omitted = text to image.",
    )
    count: Count = 1
    aspect_ratio: Optional[KleinRatio] = Field(
        default=None,
        description=(
            "Output shape. Omitted = the ratio here closest to the first image's, "
            "or `1:1` without images."
        ),
    )
    size: KleinSize = Field(
        default="1K",
        description=(
            "Size tier. With `aspect_ratio`, it fixes the output size: about 1 MP "
            "at `1K` (1024×1024 for `1:1`), twice the sides at `2K`."
        ),
    )
    format: ImageFormat = "png"
    seed: Optional[Seed] = None

    def output_size(self) -> tuple[int, int]:
        sizes = KLEIN_SIZES[self.size]
        if self.aspect_ratio is not None:
            ratio = self.aspect_ratio
        elif self.images:
            ratio = closest_ratio(*self._images["images"][0].size, sizes)
        else:
            ratio = "1:1"
        return sizes[ratio]

    @classmethod
    def warmup_sizes(cls) -> list[tuple[int, int]]:
        return [size for sizes in KLEIN_SIZES.values() for size in sizes.values()]
