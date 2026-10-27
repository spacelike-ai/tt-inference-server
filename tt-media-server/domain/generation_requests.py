# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from config.constants import ModelRunners
from domain.flux2_klein_4b_request import Flux2klein4bRequest
from domain.generation_request import GenerationRequest
from domain.ideogram4_request import Ideogram4Request
from domain.ltx23_distilled_request import Ltx23DistilledRequest
from domain.qwen_image_edit_2511_request import QwenImageEdit2511Request
from domain.wan22_i2v_a14b_request import Wan22i2vA14bRequest

# The request each generation runner consumes, keyed by runner name.
GENERATION_REQUESTS: dict[str, type[GenerationRequest]] = {
    ModelRunners.TT_WAN_2_2_I2V.value: Wan22i2vA14bRequest,
    ModelRunners.TT_QWEN_IMAGE_EDIT_2511.value: QwenImageEdit2511Request,
    ModelRunners.TT_FLUX_2_KLEIN_4B.value: Flux2klein4bRequest,
    ModelRunners.TT_IDEOGRAM_4.value: Ideogram4Request,
    ModelRunners.TT_LTX_2_3_DISTILLED.value: Ltx23DistilledRequest,
}
