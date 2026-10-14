# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import Optional

from config.settings import get_settings
from domain.image_to_image_request import ImageToImageRequest
from pydantic import Field, field_validator, model_validator

# Runners whose edit path requires a mask (mask-based inpainting). Other runners
# (e.g. FLUX.1-Kontext, which edits by instruction only) accept a missing mask.
_MASK_REQUIRED_RUNNERS = {"tt-sdxl-edit", "tt-qwen-image", "tt-qwen-image-2512"}
# Runners whose inpainting skips the first (1 - strength) of the denoising steps,
# rounded as diffusers does, and rejects a strength that leaves none.
_STRENGTH_SKIPS_STEPS_RUNNERS = {
    "tt-qwen-image",
    "tt-qwen-image-2512",
    "tt-qwen-image-edit",
    "tt-qwen-image-edit-2509",
    "tt-qwen-image-edit-2511",
}
# Runners whose masked edit generates at the configured resolution nearest the
# input image's aspect ratio, so the request can't choose a size.
_MASKED_EDIT_FIXED_SIZE_RUNNERS = {
    "tt-qwen-image-edit",
    "tt-qwen-image-edit-2509",
    "tt-qwen-image-edit-2511",
}


class ImageEditRequest(ImageToImageRequest):
    # Optional so the shared /edits endpoint serves both mask-based edits (SDXL)
    # and instruction-only edits with no mask (FLUX.1-Kontext). validate_default
    # so the per-runner check below also fires when mask is omitted entirely.
    mask: Optional[str] = Field(default=None, validate_default=True)

    @field_validator("mask")
    @classmethod
    def _require_mask_when_runner_needs_it(cls, v):
        # Reject a missing mask at validation time (clean 422) for runners that
        # require one, instead of letting it fail later in the runner's mask
        # preprocessing (500). Kontext and other runners still allow None.
        if v is None and get_settings().model_runner in _MASK_REQUIRED_RUNNERS:
            raise ValueError(f"mask is required for {get_settings().model_runner}")
        return v

    @model_validator(mode="after")
    def _check_inpainting_parameters(self):
        model_runner = get_settings().model_runner
        if self.mask is not None and model_runner in _STRENGTH_SKIPS_STEPS_RUNNERS:
            steps = self.num_inference_steps
            if self.strength is None or (
                steps is not None
                and int(steps - min(steps * self.strength, steps)) >= steps
            ):
                raise ValueError(
                    f"strength {self.strength} leaves no denoising steps for "
                    f"{model_runner}"
                )
        if (
            self.mask is not None
            and self.width is not None
            and model_runner in _MASKED_EDIT_FIXED_SIZE_RUNNERS
        ):
            raise ValueError(
                f"{model_runner} edits with a mask at the configured resolution "
                "nearest the input image's aspect ratio; omit width and height"
            )
        return self
