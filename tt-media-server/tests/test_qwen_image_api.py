# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Unit tests for the Qwen-Image media API surface.

Qwen-Image and Qwen-Image-2512 serve text-to-image on /generations and
inpainting on /edits, where the mask is required. Qwen-Image-Edit, -2509 and
-2511 serve /edits only, with an optional mask. Device and pipeline code
(dit_runners / tt_dit) runs on hardware, not here."""

import base64
import importlib.util
import pathlib
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from config.constants import DeviceTypes, ModelConfigs, ModelRunners
from domain.image_edit_request import ImageEditRequest
from domain.image_generate_request import ImageGenerateRequest

_B64 = base64.b64encode(b"fake-png-bytes").decode()

_TEXT_TO_IMAGE_RUNNERS = ["tt-qwen-image", "tt-qwen-image-2512"]
_EDIT_RUNNERS = [
    "tt-qwen-image-edit",
    "tt-qwen-image-edit-2509",
    "tt-qwen-image-edit-2511",
]
_BH_DEVICES = [DeviceTypes.P300X2, DeviceTypes.P150X8, DeviceTypes.BLACKHOLE_GALAXY]
_WH_DEVICES = [DeviceTypes.T3K, DeviceTypes.GALAXY]


def _set_runner(monkeypatch, runner: str) -> None:
    import domain.image_edit_request as edit_mod
    import domain.image_generate_request as generate_mod

    for mod in (edit_mod, generate_mod):
        monkeypatch.setattr(
            mod, "get_settings", lambda: SimpleNamespace(model_runner=runner)
        )


def _routes(monkeypatch, runner: str) -> set[str]:
    from config.settings import settings

    monkeypatch.setattr(settings, "model_runner", runner)
    path = pathlib.Path(__file__).resolve().parents[1] / "open_ai_api" / "image.py"
    spec = importlib.util.spec_from_file_location("qwen_image_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    app = FastAPI()
    app.include_router(mod.router)
    return set(app.openapi()["paths"])


@pytest.mark.parametrize("runner", _TEXT_TO_IMAGE_RUNNERS)
def test_inpainting_requires_mask(monkeypatch, runner):
    _set_runner(monkeypatch, runner)
    with pytest.raises(ValidationError):
        ImageEditRequest(prompt="a cat", image=_B64)
    request = ImageEditRequest(prompt="a cat", image=_B64, mask=_B64, strength=0.8)
    assert request.mask == _B64 and request.strength == 0.8


@pytest.mark.parametrize("runner", _EDIT_RUNNERS)
def test_edit_mask_is_optional(monkeypatch, runner):
    _set_runner(monkeypatch, runner)
    assert ImageEditRequest(prompt="add a hat", image=_B64).mask is None
    assert ImageEditRequest(prompt="add a hat", image=_B64, mask=_B64).mask == _B64


@pytest.mark.parametrize("runner", _TEXT_TO_IMAGE_RUNNERS + _EDIT_RUNNERS)
@pytest.mark.parametrize("strength", [None, 0.0, 1e-17])
def test_inpainting_rejects_strength_without_steps(monkeypatch, runner, strength):
    _set_runner(monkeypatch, runner)
    with pytest.raises(ValidationError):
        ImageEditRequest(prompt="a cat", image=_B64, mask=_B64, strength=strength)


@pytest.mark.parametrize("runner", _EDIT_RUNNERS)
def test_edit_without_mask_ignores_strength(monkeypatch, runner):
    _set_runner(monkeypatch, runner)
    assert ImageEditRequest(prompt="add a hat", image=_B64, strength=None).mask is None


@pytest.mark.parametrize(
    "runner, size, valid",
    [
        ("tt-qwen-image", (1664, 928), True),
        ("tt-qwen-image", (1344, 1344), True),
        ("tt-qwen-image", (1664, 1664), False),
        ("tt-qwen-image", (1376, 768), False),
        ("tt-qwen-image-edit-2511", (1376, 768), True),
        ("tt-qwen-image-edit-2511", (1664, 928), False),
    ],
)
def test_request_validates_size_per_runner(monkeypatch, runner, size, valid):
    _set_runner(monkeypatch, runner)
    width, height = size
    if valid:
        ImageGenerateRequest(prompt="a cat", width=width, height=height)
    else:
        with pytest.raises(ValidationError):
            ImageGenerateRequest(prompt="a cat", width=width, height=height)


@pytest.mark.parametrize("runner", _EDIT_RUNNERS)
def test_masked_edit_rejects_size(monkeypatch, runner):
    _set_runner(monkeypatch, runner)
    with pytest.raises(ValidationError):
        ImageEditRequest(
            prompt="add a hat", image=_B64, mask=_B64, width=1376, height=768
        )
    assert ImageEditRequest(prompt="add a hat", image=_B64, width=1376, height=768)


def test_segment_requests(monkeypatch):
    from model_services.image_service import ImageService

    _set_runner(monkeypatch, "tt-qwen-image")
    request = ImageEditRequest(
        prompt="a cat", image=_B64, mask=_B64, number_of_images=2, seed=3
    )
    request._segments = [0, 1]
    fields_set = set(request.model_fields_set)
    segments = [
        ImageService.create_segment_request(None, request, i, i) for i in range(2)
    ]
    # Each segment is a request of its own, waited for under its own task id.
    assert len({request._task_id} | {s._task_id for s in segments}) == 3
    # Setting a field on one segment marks it set on that segment only.
    segments[0].guidance_scale = 2.0
    assert request.model_fields_set == fields_set
    assert "guidance_scale" not in segments[1].model_fields_set
    for i, segment in enumerate(segments[1:], start=1):
        assert type(segment) is ImageEditRequest
        assert segment._segments is None and segment._start_event is None
        assert "guidance_scale" not in segment.model_fields_set
        assert (segment.image, segment.mask, segment.strength) == (_B64, _B64, 0.6)
        assert (segment.number_of_images, segment.seed) == (1, 3 + i)


@pytest.mark.parametrize("runner", _TEXT_TO_IMAGE_RUNNERS)
def test_models_schema_lists_inpainting_fields(runner):
    from open_ai_api.models import MODEL_RUNNER_TO_REQUEST_MAP

    properties = MODEL_RUNNER_TO_REQUEST_MAP[runner].model_json_schema()["properties"]
    assert {"guidance_scale", "image", "mask", "strength"} <= set(properties)


def test_request_rejects_sizes_above_1664():
    with pytest.raises(ValidationError):
        ImageGenerateRequest(prompt="a cat", width=1696, height=928)
    with pytest.raises(ValidationError):
        ImageGenerateRequest(prompt="a cat", width=928, height=1696)


@pytest.mark.parametrize("runner", _TEXT_TO_IMAGE_RUNNERS)
def test_text_to_image_serves_generations_and_edits(monkeypatch, runner):
    assert _routes(monkeypatch, runner) == {"/generations", "/edits"}


@pytest.mark.parametrize("runner", _EDIT_RUNNERS)
def test_edit_serves_edits_only(monkeypatch, runner):
    assert _routes(monkeypatch, runner) == {"/edits"}


@pytest.mark.parametrize(
    "runner, device",
    [(r, d) for r in _TEXT_TO_IMAGE_RUNNERS for d in _BH_DEVICES + _WH_DEVICES]
    + [(r, d) for r in _EDIT_RUNNERS for d in _BH_DEVICES],
)
def test_device_configs(runner, device):
    config = ModelConfigs[(ModelRunners(runner), device)]
    expected = {
        DeviceTypes.P300X2: (2, 2),
        DeviceTypes.T3K: (2, 4),
        DeviceTypes.P150X8: (2, 4),
    }.get(device, (4, 8))
    assert config["device_mesh_shape"] == expected
    assert config["max_batch_size"] == 1


@pytest.mark.parametrize("runner", _EDIT_RUNNERS)
@pytest.mark.parametrize("device", _WH_DEVICES)
def test_edit_runners_are_blackhole_only(runner, device):
    assert (ModelRunners(runner), device) not in ModelConfigs
