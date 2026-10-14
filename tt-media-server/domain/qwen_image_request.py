# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from typing import Optional

from domain.image_edit_request import ImageEditRequest


# The schema /v1/models advertises for Qwen-Image and Qwen-Image-2512: the
# fields of /edits, whose image (required there) /generations does not take.
# Requests are validated by the endpoints' own models, not this one.
class QwenImageRequest(ImageEditRequest):
    image: Optional[str] = None
