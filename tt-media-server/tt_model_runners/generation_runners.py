# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Runners of the generations API. Each consumes its model's request from
domain/ directly, with the inputs the service resolved before queueing it:
decoded images, output size, seed and the prompt the model gets.

The tt_dit pipelines take the output size per call, so one deployment serves
every size its request schema allows. Warmup runs every size and input
combination the pipeline does not warm up itself, so none compiles after the
server reports ready.

The parameters the API leaves out (steps, guidance, sampler shift) are the
models' reference defaults.
"""

import os

from config.constants import WAN22_NUM_FRAMES, Resolution
from config.settings import get_settings
from domain.flux2_klein_4b_request import KLEIN_STEPS, Flux2klein4bRequest
from domain.generation_request import GenerationRequest
from domain.ideogram4_request import IDEOGRAM_PRESETS, Ideogram4Request
from domain.qwen_image_edit_2511_request import QwenImageEdit2511Request
from domain.wan22_i2v_a14b_request import Wan22i2vA14bRequest
from huggingface_hub import snapshot_download
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt, WanPipelineI2V
from PIL import Image, ImageOps
from telemetry.image_metrics import ImageStageRecorder, sampler_name
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.dit_runners import (
    TTDiTRunner,
    TTQwenImageRunner,
    TTWan22I2VRunner,
    dit_runner_log_map,
)
from utils.decorators import log_execution_time
from utils.logger import log_exception_chain

_WARMUP_PROMPT = "Sunrise on a beach"
# Warmup only compiles and traces, so it runs fewer steps where it can.
_WARMUP_STEPS = 2

WAN_STEPS = 40
WAN_GUIDANCE_SCALES = (3.5, 3.5)
WAN_SHIFT = 5.0
QWEN_STEPS = 40
QWEN_GUIDANCE_SCALE = 4.0

_inference_timer = log_execution_time(
    f"{dit_runner_log_map[get_settings().model_runner]} inference",
    TelemetryEvent.MODEL_INFERENCE,
    os.environ.get("TT_VISIBLE_DEVICES"),
)


def _warmup_request(
    request: GenerationRequest,
    width: int,
    height: int,
    images: dict[str, list[Image.Image]] | None = None,
) -> GenerationRequest:
    request._width, request._height = width, height
    request._images = images or {}
    request._prompt = _WARMUP_PROMPT
    return request


def _weights_dir(model_weights_path: str) -> str:
    # MODEL_WEIGHTS_DIR is a local directory; otherwise the path is a Hugging
    # Face repo ID.
    if os.path.isdir(model_weights_path):
        return model_weights_path
    return snapshot_download(model_weights_path)


class _GenerationRunner(TTDiTRunner):
    def _steps(self, steps: int) -> int:
        return _WARMUP_STEPS if self._warming_up else steps

    def _image_stage_recorder(self) -> ImageStageRecorder | None:
        if self._warming_up:
            return None
        return ImageStageRecorder(
            model_type=self.settings.model_runner,
            device_id=self.device_id,
            sampler=sampler_name(self.pipeline),
            batch=1,
        )


class TTWan22I2VA14BRunner(_GenerationRunner, TTWan22I2VRunner):
    def create_pipeline(self):
        # The mesh's preset decides the sizes; the pipeline rejects the others
        # per call.
        try:
            return WanPipelineI2V.create_pipeline(
                mesh_device=self.ttnn_device, num_frames=WAN22_NUM_FRAMES
            )
        except Exception as e:
            log_exception_chain(
                self.logger, self.device_id, "Wan I2V pipeline creation failed", e
            )
            raise

    def warmup_requests(self) -> list[GenerationRequest]:
        # The pipeline's own warmup is untraced and first frame only. Each size
        # is traced here, with and without the last frame: conditioning it makes
        # the VAE encode all frames, which compiles anew.
        supported = set(self.pipeline.resolutions)
        return [
            _warmup_request(
                Wan22i2vA14bRequest.model_construct(
                    prompt=_WARMUP_PROMPT,
                    negative_prompt=_WARMUP_PROMPT,
                    last_frame="warmup" if last_frame else None,
                    frames=WAN22_NUM_FRAMES,
                ),
                width,
                height,
                {
                    name: [Image.new("RGB", (width, height))]
                    for name in ("first_frame", "last_frame")[: 1 + last_frame]
                },
            )
            for width, height in Wan22i2vA14bRequest.warmup_sizes()
            if (width, height) in supported
            for last_frame in (False, True)
        ]

    @_inference_timer
    def run(self, requests: list[Wan22i2vA14bRequest]):
        request = requests[0]
        size = (request._width, request._height)
        # Fitted to the output size: resized and center-cropped.
        image_prompt = [
            ImagePrompt(
                image=ImageOps.fit(request._images["first_frame"][0], size),
                frame_pos=0,
            )
        ]
        if request.last_frame is not None:
            image_prompt.append(
                ImagePrompt(
                    image=ImageOps.fit(request._images["last_frame"][0], size),
                    frame_pos=request.frames - 1,
                )
            )
        recorder = self._video_stage_recorder(
            Resolution(height=request._height, width=request._width)
        )
        frames = None
        try:
            frames = self.pipeline(
                prompts=[request._prompt],
                negative_prompts=[request.negative_prompt],
                image_prompt=image_prompt,
                num_inference_steps=self._steps(WAN_STEPS),
                guidance_scale=WAN_GUIDANCE_SCALES[0],
                guidance_scale_2=WAN_GUIDANCE_SCALES[1],
                flow_shift=WAN_SHIFT,
                seed=request._seed,
                height=request._height,
                width=request._width,
                num_frames=request.frames,
                traced=True,
                on_event=recorder,
            )
        finally:
            if recorder is not None:
                recorder.flush(frames)
        return frames


class TTQwenImageEdit2511Runner(_GenerationRunner, TTQwenImageRunner):
    def get_pipeline_device_params(self):
        # Per-chip trace region for every resolution of the mesh's preset, from
        # tt-metal's models/tt_dit/tests/models/qwenimage/device_params.py.
        return {"l1_small_size": 32768, "trace_region_size": 790_000_000}

    def warmup_requests(self) -> list[GenerationRequest]:
        # The pipeline compiles and traces every resolution of its preset when
        # it is created.
        return []

    @_inference_timer
    def run(self, requests: list[QwenImageEdit2511Request]):
        request = requests[0]
        recorder = self._image_stage_recorder()
        images = self.pipeline(
            prompts=[request._prompt],
            negative_prompts=[request.negative_prompt],
            image=request._images["images"],
            num_inference_steps=QWEN_STEPS,
            cfg_scale=QWEN_GUIDANCE_SCALE,
            seed=request._seed,
            width=request._width,
            height=request._height,
            on_event=recorder,
        )
        if recorder is not None:
            recorder.flush(images)
        return images


class TTFlux2Klein4BRunner(_GenerationRunner):
    def create_pipeline(self):
        from models.tt_dit.pipelines.flux2.pipeline_flux2 import Flux2Pipeline

        try:
            return Flux2Pipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                checkpoint_name=self.settings.model_weights_path,
            )
        except Exception as e:
            log_exception_chain(
                self.logger, self.device_id, "FLUX.2 pipeline creation failed", e
            )
            raise

    def get_pipeline_device_params(self):
        # From tt-metal's models/tt_dit/tests/models/flux2/device_params.py: the
        # VAE's conv2d needs L1_SMALL, the denoising loop is traced.
        return {"l1_small_size": 65536, "trace_region_size": 31_000_000}

    def warmup_requests(self) -> list[GenerationRequest]:
        # Text to image only: which input-image sequence lengths compile anew
        # depends on the pipeline, which doesn't take images yet.
        return [
            _warmup_request(
                Flux2klein4bRequest.model_construct(prompt=_WARMUP_PROMPT),
                width,
                height,
            )
            for width, height in Flux2klein4bRequest.warmup_sizes()
        ]

    @_inference_timer
    def run(self, requests: list[Flux2klein4bRequest]):
        request = requests[0]
        recorder = self._image_stage_recorder()
        images = self.pipeline(
            prompts=[request._prompt],
            images=request._images.get("images") or None,
            num_inference_steps=self._steps(KLEIN_STEPS),
            seed=request._seed,
            width=request._width,
            height=request._height,
            traced=True,
            on_event=recorder,
        )
        if recorder is not None:
            recorder.flush(images)
        return images


class TTIdeogram4Runner(_GenerationRunner):
    def create_pipeline(self):
        from models.tt_dit.pipelines.ideogram4.pipeline_ideogram4 import (
            Ideogram4Pipeline,
        )

        try:
            return Ideogram4Pipeline.create_pipeline(
                mesh_device=self.ttnn_device,
                weights_dir=_weights_dir(self.settings.model_weights_path),
            )
        except Exception as e:
            log_exception_chain(
                self.logger, self.device_id, "Ideogram 4 pipeline creation failed", e
            )
            raise

    def get_pipeline_device_params(self):
        # From tt-metal's models/tt_dit/tests/models/ideogram4/test_pipeline_class_ideogram4.py.
        return {"l1_small_size": 32768, "trace_region_size": 60_000_000}

    def warmup_requests(self) -> list[GenerationRequest]:
        # The presets differ only in step count and the per-step guidance
        # weight, an input of the same step program, and the pipeline captures
        # its trace anew on every call. So the fastest preset compiles every
        # size for all three; it fixes its step count, so warmup can't shorten it.
        return [
            _warmup_request(
                Ideogram4Request.model_construct(rendering_speed="turbo"), width, height
            )
            for width, height in Ideogram4Request.warmup_sizes()
        ]

    @_inference_timer
    def run(self, requests: list[Ideogram4Request]):
        request = requests[0]
        recorder = self._image_stage_recorder()
        images = self.pipeline(
            prompts=[request._prompt],
            preset=IDEOGRAM_PRESETS[request.rendering_speed],
            seed=request._seed,
            width=request._width,
            height=request._height,
            traced=True,
            on_event=recorder,
        )
        if recorder is not None:
            recorder.flush(images)
        return images
