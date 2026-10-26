# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""What the generations API's runners pass their tt_dit pipelines, and the
warmup requests they build. The pipelines are stubs; the runner module is
loaded against fake tt-metal modules, as in test_video_stage_metrics_wiring."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from domain.flux2_klein_4b_request import Flux2klein4bRequest
from domain.ideogram4_request import Ideogram4Request
from domain.qwen_image_edit_2511_request import QwenImageEdit2511Request
from domain.wan22_i2v_a14b_request import (
    WAN_NEGATIVE_PROMPT,
    WAN_SIZES,
    Wan22i2vA14bRequest,
)
from PIL import Image

_RUNNERS = Path(__file__).resolve().parent.parent / "tt_model_runners"

_TT_METAL_MODULES = (
    "models",
    "models.common",
    "models.common.utility_functions",
    "models.tt_dit",
    "models.tt_dit.pipelines",
    "models.tt_dit.pipelines.flux1",
    "models.tt_dit.pipelines.flux1.pipeline_flux1",
    "models.tt_dit.pipelines.flux1.pipeline_flux1_kontext",
    "models.tt_dit.pipelines.minimax_h3",
    "models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3",
    "models.tt_dit.pipelines.mochi",
    "models.tt_dit.pipelines.mochi.pipeline_mochi",
    "models.tt_dit.pipelines.motif",
    "models.tt_dit.pipelines.motif.pipeline_motif",
    "models.tt_dit.pipelines.qwenimage",
    "models.tt_dit.pipelines.qwenimage.pipeline_qwenimage",
    "models.tt_dit.pipelines.stable_diffusion_35_large",
    "models.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large",
    "models.tt_dit.pipelines.wan",
    "models.tt_dit.pipelines.wan.pipeline_wan",
    "models.tt_dit.pipelines.wan.pipeline_wan_i2v",
)


def _load_generation_runners():
    settings = SimpleNamespace(
        model_runner="tt-ideogram-4",
        model_service="generation",
        device_mesh_shape=(2, 4),
        enable_telemetry=False,
    )
    settings_module = ModuleType("config.settings")
    settings_module.settings = settings
    settings_module.get_settings = lambda: settings

    replaced = {
        name: sys.modules.get(name)
        for name in (
            *_TT_METAL_MODULES,
            "ttnn",
            "config.settings",
            "tt_model_runners.dit_runners",
        )
    }
    logger_module = sys.modules.get("utils.logger")
    patch_logger = logger_module is not None and not hasattr(
        logger_module, "log_exception_chain"
    )
    for name in _TT_METAL_MODULES:
        sys.modules.setdefault(name, Mock())
    sys.modules.setdefault("ttnn", Mock())
    sys.modules["config.settings"] = settings_module
    if patch_logger:
        logger_module.log_exception_chain = Mock()
    try:
        modules = {}
        for name in ("dit_runners", "generation_runners"):
            spec = importlib.util.spec_from_file_location(
                f"tt_model_runners.{name}", _RUNNERS / f"{name}.py"
            )
            modules[name] = importlib.util.module_from_spec(spec)
            sys.modules[f"tt_model_runners.{name}"] = modules[name]
            spec.loader.exec_module(modules[name])
        return modules["generation_runners"]
    finally:
        for name, module in replaced.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.modules.pop("tt_model_runners.generation_runners", None)
        if patch_logger:
            del logger_module.log_exception_chain


runners = _load_generation_runners()


class _Pipeline:
    def __init__(self):
        self.calls = []
        self.resolutions = [
            size for sizes in WAN_SIZES.values() for size in sizes.values()
        ]

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return ["output"]


def _runner(class_name: str, warming_up: bool = False):
    cls = getattr(runners, class_name)
    runner = cls.__new__(cls)
    runner.device_id = "0"
    runner._warming_up = warming_up
    runner._image_stage_recorder = lambda: None
    runner._video_stage_recorder = lambda resolution: None
    runner.logger = Mock()
    runner.settings = SimpleNamespace(model_runner="tt-ideogram-4")
    runner.pipeline = _Pipeline()
    return runner


def _resolved(request, width, height, images=None, seed=5, prompt=None):
    request._width, request._height, request._seed = width, height, seed
    request._images = images or {}
    request._prompt = prompt if prompt is not None else request.prompt
    return request


def test_wan_passes_both_frames_fitted_to_the_output(monkeypatch):
    monkeypatch.setattr(
        runners, "ImagePrompt", lambda image, frame_pos: (image, frame_pos)
    )
    runner = _runner("TTWan22I2VA14BRunner")
    # A wide first frame is center-cropped, not stretched.
    first = Image.new("RGB", (2000, 720), "red")
    first.paste((0, 0, 255), (0, 0, 360, 720))
    last = Image.new("RGB", (640, 360))
    request = _resolved(
        Wan22i2vA14bRequest(
            prompt="a boat",
            first_frame="https://x/a.png",
            last_frame="https://x/b.png",
            aspect_ratio="16:9",
            size="720p",
        ),
        1280,
        720,
        {"first_frame": [first], "last_frame": [last]},
    )
    assert runner.run([request]) == ["output"]
    (call,) = runner.pipeline.calls
    (fitted_first, first_pos), (fitted_last, last_pos) = call.pop("image_prompt")
    assert (first_pos, last_pos) == (0, 80)
    assert fitted_first.size == fitted_last.size == (1280, 720)
    assert fitted_first.getpixel((0, 0)) == (255, 0, 0)
    assert call == {
        "prompts": ["a boat"],
        "negative_prompts": [WAN_NEGATIVE_PROMPT],
        "num_inference_steps": 40,
        "guidance_scale": 3.5,
        "guidance_scale_2": 3.5,
        "flow_shift": 5.0,
        "seed": 5,
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "traced": True,
        "on_event": None,
    }


def test_qwen_passes_the_input_images():
    runner = _runner("TTQwenImageEdit2511Runner")
    images = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
    request = _resolved(
        QwenImageEdit2511Request(
            prompt="add a hat", negative_prompt="blurry", images=["https://x/a.png"] * 2
        ),
        1376,
        768,
        {"images": images},
    )
    runner.run([request])
    (call,) = runner.pipeline.calls
    assert call["image"] is images
    assert (call["width"], call["height"], call["cfg_scale"]) == (1376, 768, 4.0)
    assert call["negative_prompts"] == ["blurry"]
    assert call["num_inference_steps"] == 40


@pytest.mark.parametrize("with_images", [False, True])
def test_klein_edits_only_with_images(with_images):
    runner = _runner("TTFlux2Klein4BRunner")
    images = {"images": [Image.new("RGB", (8, 8))]} if with_images else {}
    request = _resolved(Flux2klein4bRequest(prompt="a cat"), 1024, 1024, images)
    runner.run([request])
    (call,) = runner.pipeline.calls
    assert call["images"] == (images["images"] if with_images else None)
    assert call["num_inference_steps"] == 4


def test_ideogram_gets_the_caption_and_preset():
    runner = _runner("TTIdeogram4Runner")
    request = Ideogram4Request(
        structured_prompt={
            "compositional_deconstruction": {"background": "b", "elements": []}
        },
        rendering_speed="quality",
    )
    request = _resolved(request, 1024, 1024, prompt=request.model_prompt())
    runner.run([request])
    (call,) = runner.pipeline.calls
    assert call["prompts"] == [
        '{"compositional_deconstruction":{"background":"b","elements":[]}}'
    ]
    assert call["preset"] == "V4_QUALITY_48"


@pytest.mark.parametrize(
    ("class_name", "combinations"),
    [
        # 4 sizes, with and without the last frame.
        ("TTWan22I2VA14BRunner", 8),
        ("TTFlux2Klein4BRunner", 14),
        # 14 sizes; one preset compiles all three.
        ("TTIdeogram4Runner", 14),
    ],
)
def test_warmup_covers_every_combination(class_name, combinations):
    runner = _runner(class_name, warming_up=True)
    requests = runner.warmup_requests()
    assert len(requests) == combinations
    for request in requests:
        runner.pipeline.calls.clear()
        runner.run([request])
        (call,) = runner.pipeline.calls
        assert (call["width"], call["height"]) == (request._width, request._height)
        if class_name != "TTIdeogram4Runner":
            assert call["num_inference_steps"] == 2


def test_wan_warmup_conditions_the_last_frame(monkeypatch):
    monkeypatch.setattr(
        runners, "ImagePrompt", lambda image, frame_pos: (image, frame_pos)
    )
    runner = _runner("TTWan22I2VA14BRunner", warming_up=True)
    runner.pipeline.resolutions = list(WAN_SIZES["480p"].values())
    requests = runner.warmup_requests()
    assert {(r._width, r._height) for r in requests} == set(WAN_SIZES["480p"].values())
    frame_positions = set()
    for request in requests:
        runner.pipeline.calls.clear()
        runner.run([request])
        (call,) = runner.pipeline.calls
        frame_positions.add(tuple(pos for _, pos in call["image_prompt"]))
    assert frame_positions == {(0,), (0, 80)}


def test_qwen_pipeline_warms_itself_up():
    assert _runner("TTQwenImageEdit2511Runner").warmup_requests() == []


def test_ideogram_weights_are_a_local_directory_or_a_repo(monkeypatch, tmp_path):
    monkeypatch.setattr(runners, "snapshot_download", lambda repo: f"/hub/{repo}")
    assert runners._weights_dir(str(tmp_path)) == str(tmp_path)
    assert (
        runners._weights_dir("ideogram-ai/ideogram-4") == "/hub/ideogram-ai/ideogram-4"
    )
