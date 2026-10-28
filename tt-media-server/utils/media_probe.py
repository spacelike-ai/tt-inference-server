# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import io
from dataclasses import dataclass

from PIL import Image, ImageOps


@dataclass(frozen=True)
class ReferenceMediaProbe:
    """Container metadata of one video or audio clip.

    ``size`` (width, height) and ``fps`` are those of the first video stream, ``audio_channels``
    that of the first audio stream; each is None when the stream is absent.
    """

    duration_s: float
    size: tuple[int, int] | None
    fps: float | None
    audio_channels: int | None


def decode_image(data: bytes) -> Image.Image:
    """The RGB image in ``data``, upright as a camera's EXIF orientation says.

    Raises ``ValueError`` if it has more than ``Image.MAX_IMAGE_PIXELS``
    pixels, and PIL's ``UnidentifiedImageError`` or ``OSError`` if it is not a
    decodable image.
    """
    too_large = ValueError(f"image has more than {Image.MAX_IMAGE_PIXELS} pixels")
    try:
        image = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise too_large
    # Checked before decoding: PIL only refuses twice the limit.
    if Image.MAX_IMAGE_PIXELS and image.width * image.height > Image.MAX_IMAGE_PIXELS:
        raise too_large
    image.load()
    return ImageOps.exif_transpose(image).convert("RGB")


def probe_reference_media(raw: bytes) -> ReferenceMediaProbe:
    """Probe encoded audio/video bytes without decoding them.

    Uses PyAV's container duration (microseconds) and the frame rate a decoder reads (average,
    else guessed). Raises ``ValueError`` when the bytes are not a readable audio/video container or
    have no duration.
    """
    import av

    try:
        with av.open(io.BytesIO(raw)) as container:
            if container.duration is None or container.duration <= 0:
                raise ValueError("media container reports no duration")
            size = fps = audio_channels = None
            if container.streams.video:
                stream = container.streams.video[0]
                size = (stream.codec_context.width, stream.codec_context.height)
                rate = stream.average_rate or stream.guessed_rate
                fps = float(rate) if rate else None
            if container.streams.audio:
                audio_channels = len(container.streams.audio[0].layout.channels)
            return ReferenceMediaProbe(
                duration_s=container.duration / 1_000_000,
                size=size,
                fps=fps,
                audio_channels=audio_channels,
            )
    except av.FFmpegError as exc:  # av.AVError was removed in PyAV 14
        raise ValueError("not a readable video or audio file") from exc
