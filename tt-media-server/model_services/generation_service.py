# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import asyncio
import base64
import binascii
import contextlib
import hashlib
import json
import secrets
import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from config.constants import JobTypes, generation_job_dir
from config.settings import settings
from domain.generation_request import MAX_SEED, OUTPUT_URL, GenerationRequest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from model_services.base_job_service import BaseJobService
from PIL import Image, ImageOps, UnidentifiedImageError
from utils.media_downloader import (
    MediaDownloadFetchError,
    MediaDownloadPolicyError,
    MediaDownloadTooLargeError,
    download_media_url,
)
from utils.video_manager import VideoAudioResult, VideoManager, probe_video

# The file in a job's directory listing its outputs.
MANIFEST = "outputs.json"
# Per input image, inline or fetched.
MEDIA_MAX_BYTES = 30_000_000

_IMAGE_FORMATS = {
    "png": ("PNG", "image/png"),
    "jpeg": ("JPEG", "image/jpeg"),
    "webp": ("WEBP", "image/webp"),
}


def _too_large() -> HTTPException:
    return HTTPException(
        status_code=413, detail=f"An image is over {MEDIA_MAX_BYTES} bytes"
    )


def _decode_data_uri(uri: str) -> bytes:
    data = base64.b64decode(uri.split(",", 1)[1], validate=True)
    if len(data) > MEDIA_MAX_BYTES:
        raise _too_large()
    return data


def _decode_image(data: bytes) -> Image.Image:
    too_large = ValueError(f"image has more than {Image.MAX_IMAGE_PIXELS} pixels")
    try:
        image = Image.open(BytesIO(data))
    except Image.DecompressionBombError:
        raise too_large
    # Checked before decoding: PIL only refuses twice the limit.
    if Image.MAX_IMAGE_PIXELS and image.width * image.height > Image.MAX_IMAGE_PIXELS:
        raise too_large
    image.load()
    # Upright, as camera photos with an EXIF orientation are meant to be seen.
    return ImageOps.exif_transpose(image).convert("RGB")


def _request_parameters(request: GenerationRequest) -> dict:
    """The request as stored with its job, with data URIs replaced by their
    SHA-256 so the record stays small."""
    parameters = request.model_dump(mode="json")
    for name in request.MEDIA_FIELDS:
        value = parameters.get(name)
        if isinstance(value, list):
            parameters[name] = [_digest_data_uri(item) for item in value]
        elif value is not None:
            parameters[name] = _digest_data_uri(value)
    return parameters


def _digest_data_uri(value: str) -> str:
    if value[:5].lower() != "data:":
        return value
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _write_video(result: Any, fps: int, path: Path) -> None:
    """Write a runner's video to ``path`` as a faststart MP4.

    The result is frames, frames with a soundtrack, or the path of an MP4 the
    pipeline wrote itself, which is remuxed for faststart.
    """
    if isinstance(result, VideoAudioResult):
        encoded = VideoManager().export_to_mp4_with_audio(
            result.frames,
            result.audio,
            result.sampling_rate,
            fps=result.fps,
            pixel_format=result.pixel_format,
        )
    elif isinstance(result, str):
        try:
            VideoManager.ensure_faststart(result, str(path))
        finally:
            Path(result).unlink(missing_ok=True)
        return
    else:
        # The encoder already writes faststart MP4s.
        encoded = VideoManager().export_to_mp4(result, fps=fps)
    shutil.move(encoded, path)


def read_manifest(result_path: str) -> list[dict]:
    """The outputs of a completed job, from the directory it wrote them to."""
    return json.loads((Path(result_path) / MANIFEST).read_text())


# Serves the generations API: one job per submitted request, producing an image
# or a video, written to the job's own directory.
class GenerationService(BaseJobService):
    async def submit(self, request: GenerationRequest) -> dict:
        """Validate what the request schema can't, fetch its media, and queue it.

        Returns the job's public metadata.
        """
        try:
            self.scheduler.check_is_model_ready()
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="Model is not ready",
                headers={"Retry-After": "30"},
            )

        await self._load_images(request)
        request._width, request._height = request.output_size()
        request._seed = (
            request.seed
            if request.seed is not None
            else secrets.randbelow(MAX_SEED + 1)
        )

        # A cancelled queued job is dropped before a worker claims it; a running
        # one is `cancelling` until the pipeline finishes.
        request._start_event = self._createStartEvent()
        request._cancel_event = self._processManager.Event()
        request._worker_assignment = self._processManager.Namespace()
        request._worker_assignment.identity = None
        return await self._job_manager.create_job(
            job_id=request._task_id,
            job_type=JobTypes.GENERATION,
            model=request.MODEL_ID,
            request=request,
            task_function=self.process_request,
            start_event=request._start_event,
            cancel_event=request._cancel_event,
            worker_assignment=request._worker_assignment,
            request_parameters=_request_parameters(request),
            result_path=generation_job_dir(request._task_id),
        )

    async def _load_images(self, request: GenerationRequest) -> None:
        """Fetch and decode every input image, reporting all the bad ones.

        A URL the fetch policy refuses is a 400 and an image over the size
        limit a 413, as for the rest of the server's media URLs; every other
        problem is a 422.
        """
        deadline = time.monotonic() + settings.media_url_timeout_seconds
        errors = []
        for name, values in request.media().items():
            is_list = isinstance(getattr(request, name), list)
            images = []
            for index, value in enumerate(values):
                loc = ("body", name, index) if is_list else ("body", name)
                try:
                    if value.startswith("https://"):
                        data = await download_media_url(
                            value, deadline=deadline, max_bytes=MEDIA_MAX_BYTES
                        )
                    elif output_url := OUTPUT_URL.match(value):
                        data = self._read_output(*output_url.groups())
                    else:
                        data = _decode_data_uri(value)
                    images.append(await asyncio.to_thread(_decode_image, data))
                except MediaDownloadPolicyError as e:
                    raise HTTPException(status_code=400, detail=str(e))
                except MediaDownloadTooLargeError:
                    raise _too_large()
                except (MediaDownloadFetchError, binascii.Error, ValueError) as e:
                    errors.append({"loc": loc, "msg": str(e), "type": "value_error"})
                except (UnidentifiedImageError, OSError):
                    errors.append(
                        {
                            "loc": loc,
                            "msg": "not a decodable image (PNG, JPEG, WebP, ...)",
                            "type": "value_error",
                        }
                    )
            request._images[name] = images
        if errors:
            raise RequestValidationError(errors)

    def _read_output(self, job_id: str, index: str) -> bytes:
        """The file of an earlier job's output, by the parts of its `url`."""
        # A job's result path is set from submit on, so it says nothing about
        # whether the job finished.
        metadata = self.get_job_metadata(job_id)
        result_path = self.get_job_result_path(job_id)
        if (
            metadata is None
            or metadata["job_type"] != JobTypes.GENERATION.value
            or metadata["status"] != "completed"
            or result_path is None
        ):
            raise ValueError(f"job {job_id} has no completed result")
        try:
            outputs = read_manifest(result_path)
        except FileNotFoundError:
            raise ValueError(f"the outputs of job {job_id} are gone")
        if int(index) >= len(outputs):
            raise ValueError(f"job {job_id} has {len(outputs)} outputs")
        output = outputs[int(index)]
        if output["type"] != "image":
            raise ValueError(f"output {index} of job {job_id} is not an image")
        return (Path(result_path) / output["file"]).read_bytes()

    async def pre_process(self, request: GenerationRequest) -> GenerationRequest:
        request._prompt = request.model_prompt()
        return request

    async def post_process(self, result, input_request: GenerationRequest) -> str:
        """Write the output to the job's directory; returns the directory.

        ``count`` is 1, so a job has one output. More will come from one
        batched pipeline call, not from one request per output.
        """
        writing = asyncio.ensure_future(
            asyncio.to_thread(self._write_outputs, input_request, [result])
        )
        try:
            return await asyncio.shield(writing)
        except asyncio.CancelledError:
            # The thread can't be stopped. Once it is done, remove what it
            # wrote: a cancelled job has no result, so nothing else would.
            with contextlib.suppress(Exception):
                shutil.rmtree(await writing, ignore_errors=True)
            raise

    def _write_outputs(self, request: GenerationRequest, results: list[Any]) -> str:
        # A runner may already have written into it.
        result_path = generation_job_dir(request._task_id)
        directory = Path(result_path)
        directory.mkdir(parents=True, exist_ok=True)
        try:
            outputs = [
                self._write_output(directory, index, request, result)
                for index, result in enumerate(results)
            ]
            (directory / MANIFEST).write_text(json.dumps(outputs))
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        # Exactly the result path the job was created with.
        return result_path

    @staticmethod
    def _write_output(
        directory: Path, index: int, request: GenerationRequest, result: Any
    ) -> dict:
        output = {
            "type": request.OUTPUT_TYPE,
            "seed": request._seed,
            "prompt": request._prompt,
        }
        if request.OUTPUT_TYPE == "image":
            pil_format, content_type = _IMAGE_FORMATS[request.format]
            path = directory / f"{index}.{request.format}"
            image = result.convert("RGB") if pil_format == "JPEG" else result
            image.save(path, format=pil_format)
            return {
                **output,
                "file": path.name,
                "content_type": content_type,
                "width": image.width,
                "height": image.height,
            }

        path = directory / f"{index}.mp4"
        _write_video(result, request.fps, path)
        probe = probe_video(str(path))
        num_frames = probe.num_frames or request.frames
        return {
            **output,
            "file": path.name,
            "content_type": "video/mp4",
            "width": probe.width or request._width,
            "height": probe.height or request._height,
            "fps": request.fps,
            "num_frames": num_frames,
            "duration": num_frames / request.fps,
            "audio": request.AUDIO,
        }
