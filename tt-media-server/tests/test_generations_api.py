# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Unit tests for the generations API: the four models' request schemas,
checked against docs/api/generation, and submit → job → content → delete
through the router with a fake device. Runners and tt_dit run on hardware, not
here."""

import asyncio
import base64
import importlib.util
import json
import pathlib
import threading
from io import BytesIO
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import yaml
from config.constants import JobTypes
from domain.flux2_klein_4b_request import Flux2klein4bRequest
from domain.ideogram4_request import Ideogram4Request
from domain.ltx23_distilled_request import Ltx23DistilledRequest
from domain.qwen_image_edit_2511_request import QwenImageEdit2511Request
from domain.wan22_i2v_a14b_request import WAN_NEGATIVE_PROMPT, Wan22i2vA14bRequest
from fastapi import FastAPI
from PIL import Image
from pydantic import ValidationError
from utils.video_manager import VideoAudioResult, VideoManager

_SPEC_DIR = pathlib.Path(__file__).resolve().parents[1] / "docs" / "api" / "generation"

_CAPTION = {
    "compositional_deconstruction": {
        "background": "a brick wall",
        "elements": [{"type": "text", "text": "OPEN", "bbox": [10, 20, 30, 40]}],
    }
}


def _data_uri(width: int = 64, height: int = 64, image_format: str = "PNG") -> str:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, format=image_format)
    return f"data:image/{image_format.lower()};base64," + base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")


def _with_images(request, **images):
    request._images = {
        name: [Image.new("RGB", size) for size in sizes]
        for name, sizes in images.items()
    }
    return request


# --- Request schemas against the spec ----------------------------------------------


def _resolved(schema: dict, defs: dict) -> dict:
    """A property's own schema: Optional's non-null branch, $refs followed."""
    if "anyOf" in schema:
        (schema,) = [branch for branch in schema["anyOf"] if branch != {"type": "null"}]
    if "$ref" in schema and schema["$ref"].startswith("#/$defs/"):
        schema = defs[schema["$ref"].removeprefix("#/$defs/")]
    return schema


@pytest.mark.parametrize(
    "request_model",
    [
        Wan22i2vA14bRequest,
        QwenImageEdit2511Request,
        Flux2klein4bRequest,
        Ideogram4Request,
        Ltx23DistilledRequest,
    ],
)
def test_request_schema_matches_the_spec(request_model):
    spec = yaml.safe_load(
        (_SPEC_DIR / "models" / f"{request_model.MODEL_ID}.yaml").read_text()
    )
    expected = spec["components"]["schemas"][request_model.__name__]
    actual = request_model.model_json_schema()
    defs = actual.get("$defs", {})

    assert actual["additionalProperties"] is False
    assert set(actual["properties"]) == set(expected["properties"])
    assert set(actual.get("required", [])) == set(expected["required"])
    for name, want in expected["properties"].items():
        got = _resolved(actual["properties"][name], defs)
        default = actual["properties"][name].get("default")
        if "enum" in want:
            assert got.get("enum") == want["enum"], name
        if "default" in want:
            assert default == want["default"], name
        for bound in ("minimum", "maximum", "minItems", "maxItems", "minLength"):
            if bound in want:
                assert got.get(bound) == want[bound], (name, bound)


def test_ideogram_caption_matches_the_spec():
    spec = yaml.safe_load(
        (_SPEC_DIR / "models" / "ideogram-ai" / "ideogram-4.yaml").read_text()
    )["components"]["schemas"]
    defs = Ideogram4Request.model_json_schema()["$defs"]
    for name in (
        "Ideogram4Caption",
        "Ideogram4ObjectElement",
        "Ideogram4TextElement",
    ):
        assert list(defs[name]["properties"]) == list(spec[name]["properties"]), name
        assert set(defs[name].get("required", [])) == set(spec[name]["required"])


def test_unknown_field_rejected():
    with pytest.raises(ValidationError, match="extra"):
        Flux2klein4bRequest(prompt="a cat", steps=4)


@pytest.mark.parametrize(
    "media",
    ["http://example.com/a.png", "iVBORw0KGgo=", "data:text/plain;base64,aGk="],
)
def test_media_must_be_https_or_image_data_uri(media):
    with pytest.raises(ValidationError, match="https:// URL, a base64 data URI"):
        Wan22i2vA14bRequest(
            prompt="a boat", first_frame=media, aspect_ratio="16:9", size="480p"
        )


def test_wan_fields():
    request = Wan22i2vA14bRequest(
        prompt="a boat", first_frame="https://x/a.png", aspect_ratio="9:16", size="720p"
    )
    assert (request.frames, request.fps) == (81, 16)
    assert request.negative_prompt == WAN_NEGATIVE_PROMPT
    assert request.output_size() == (720, 1280)
    for fields in ({"aspect_ratio": "1:1"}, {"frames": 97}, {"aspect_ratio": None}):
        with pytest.raises(ValidationError):
            Wan22i2vA14bRequest(
                prompt="a boat",
                first_frame="https://x/a.png",
                **{"aspect_ratio": "16:9", "size": "480p", **fields},
            )
    assert len(set(Wan22i2vA14bRequest.warmup_sizes())) == 4


@pytest.mark.parametrize("fields", [{}, {"duration": 6}, {"duration": 6.12}])
def test_ltx_serves_one_shape(fields):
    request = Ltx23DistilledRequest(prompt="a violinist", **fields)
    assert (request.frames, request.fps, request.output_size()) == (
        153,
        25,
        (1920, 1088),
    )


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"duration": 5.8}, "145 frames"),
        ({"duration": float("inf")}, "finite number"),
        ({"frames": 150}, "greater than or equal to 153"),
        ({"fps": 24}, "greater than or equal to 25"),
        ({"height": 720}, "greater than or equal to 1088"),
        ({"num_inference_steps": 11}, "extra"),
    ],
)
def test_ltx_rejects(fields, message):
    with pytest.raises(ValidationError, match=message):
        Ltx23DistilledRequest(prompt="a violinist", **fields)


def test_qwen_size_follows_last_image():
    request = QwenImageEdit2511Request(
        prompt="add a hat", images=["https://x/a.png", "https://x/b.png"]
    )
    assert request.negative_prompt == " "
    _with_images(request, images=[(1000, 1000), (1600, 900)])
    assert request.output_size() == (1376, 768)
    request.aspect_ratio = "3:4"
    assert request.output_size() == (896, 1184)


@pytest.mark.parametrize(
    "fields", [{"images": ["https://x/a.png"] * 4}, {"count": 2}, {"size": "1K"}]
)
def test_qwen_rejects(fields):
    with pytest.raises(ValidationError):
        QwenImageEdit2511Request(
            prompt="x", **{"images": ["https://x/a.png"], **fields}
        )


@pytest.mark.parametrize(
    ("fields", "images", "expected"),
    [
        ({}, [], (1024, 1024)),
        ({"size": "2K"}, [], (2048, 2048)),
        ({"aspect_ratio": "16:9", "size": "2K"}, [], (2752, 1536)),
        ({}, [(1600, 900), (100, 100)], (1376, 768)),
        ({"size": "2K"}, [(900, 1250)], (1792, 2368)),
    ],
)
def test_klein_output_size(fields, images, expected):
    if images:
        fields["images"] = ["https://x/a.png"] * len(images)
        request = _with_images(
            Flux2klein4bRequest(prompt="a cat", **fields), images=images
        )
    else:
        request = Flux2klein4bRequest(prompt="a cat", **fields)
    assert request.output_size() == expected


def test_klein_has_14_sizes():
    sizes = set(Flux2klein4bRequest.warmup_sizes())
    assert len(sizes) == 14
    assert all(width % 16 == 0 and height % 16 == 0 for width, height in sizes)


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, (2048, 2048)),
        ({"size": "1K"}, (1024, 1024)),
        ({"aspect_ratio": "16:9", "size": "1K"}, (1280, 720)),
        ({"aspect_ratio": "3:2"}, (2048, 1360)),
    ],
)
def test_ideogram_output_size(fields, expected):
    request = Ideogram4Request(structured_prompt=_CAPTION, **fields)
    assert request.output_size() == expected
    assert len(set(Ideogram4Request.warmup_sizes())) == 14


def test_ideogram_caption_is_written_in_the_model_key_order():
    request = Ideogram4Request(
        structured_prompt={
            "compositional_deconstruction": {
                "elements": [{"desc": "d", "bbox": [0, 0, 1, 1], "type": "obj"}],
                "background": "b",
            },
            "style_description": {
                "medium": "m",
                "photo": "p",
                "lighting": "l",
                "aesthetics": "a",
            },
            "high_level_description": "Ein Café",
        }
    )
    assert request.model_prompt() == (
        '{"high_level_description":"Ein Café",'
        '"style_description":{"aesthetics":"a","lighting":"l","photo":"p","medium":"m"},'
        '"compositional_deconstruction":{"background":"b",'
        '"elements":[{"type":"obj","bbox":[0,0,1,1],"desc":"d"}]}}'
    )


@pytest.mark.parametrize(
    ("caption", "message"),
    [
        (
            {"compositional_deconstruction": {"background": "b", "elements": [
                {"type": "obj", "desc": "d", "bbox": [30, 0, 10, 10]}
            ]}},
            "ymin <= ymax",
        ),
        (
            {
                "style_description": {
                    "aesthetics": "a",
                    "lighting": "l",
                    "medium": "m",
                    "photo": "p",
                    "art_style": "s",
                },
                **_CAPTION,
            },
            "exactly one of photo and art_style",
        ),
        (
            {"compositional_deconstruction": {"background": "b", "elements": [
                {"type": "obj", "desc": "d", "color_palette": ["#ff0000"]}
            ]}},
            "pattern",
        ),
        ({"caption": "plain"}, "extra"),
    ],
)  # fmt: skip
def test_ideogram_rejects_invalid_captions(caption, message):
    with pytest.raises(ValidationError, match=message):
        Ideogram4Request(structured_prompt=caption)


def test_ideogram_has_no_plain_prompt():
    with pytest.raises(ValidationError, match="extra"):
        Ideogram4Request(prompt="a poster", structured_prompt=_CAPTION)


# --- Router and jobs -------------------------------------------------------------

_RETENTION_SECONDS = 3600


def _router_module(monkeypatch, runner: str):
    from config.settings import settings

    monkeypatch.setattr(settings, "model_runner", runner)
    monkeypatch.setattr(settings, "job_retention_seconds", _RETENTION_SECONDS)
    path = (
        pathlib.Path(__file__).resolve().parents[1] / "open_ai_api" / "generations.py"
    )
    spec = importlib.util.spec_from_file_location("generations_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_the_served_model_is_mounted(monkeypatch):
    module = _router_module(monkeypatch, "tt-ideogram-4")
    app = FastAPI()
    app.include_router(module.router)
    paths = set(app.openapi()["paths"])
    assert paths == {
        "/v1/generations/ideogram-ai/ideogram-4",
        "/v1/jobs",
        "/v1/jobs/{job_id}",
        "/v1/jobs/{job_id}/content/{index}",
        "/v1/jobs/{job_id}/cancel",
    }


@pytest.mark.parametrize("job_type", ["training", "video"])
def test_other_services_jobs_are_404(monkeypatch, job_type):
    # E.g. restored from a job database shared with a training deployment.
    module = _router_module(monkeypatch, "tt-ideogram-4")
    service = SimpleNamespace(
        get_job_metadata=lambda job_id: {"id": job_id, "job_type": job_type}
    )
    with pytest.raises(module.HTTPException) as error:
        module._job_or_404(service, "j")
    assert error.value.status_code == 404


class _FakeDevice:
    """Stands in for the scheduler and device: each request yields an image,
    or a video for Wan, after ``delay`` seconds. ``video``, if set, makes a
    video's result instead."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.requests = []
        self.error = None
        self.video = None

    def check_is_model_ready(self):
        return True

    async def process(self, request):
        self.requests.append(request)
        if request._start_event is not None:
            request._start_event.set()
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if request.OUTPUT_TYPE == "video" and self.video is not None:
            return self.video(request)
        if request.OUTPUT_TYPE == "video":
            return np.zeros(
                (request.frames, request._height, request._width, 3), np.uint8
            )
        return Image.new("RGB", (request._width, request._height), "blue")


@pytest.fixture
def generations_app(monkeypatch, tmp_path):
    """(build(runner, delay) -> (client, device)), sharing one job manager."""
    from model_services.generation_service import GenerationService
    from resolver.service_resolver import service_resolver
    from security.api_key_checker import get_api_key
    from utils.job_manager import JobManager

    monkeypatch.setenv("TT_GENERATIONS_DIR", str(tmp_path))
    monkeypatch.setattr(
        "utils.video_manager._VIDEO_OUTPUT_DIR", tmp_path.parent / "encoded"
    )
    monkeypatch.setattr(
        "utils.job_manager.get_settings",
        lambda: SimpleNamespace(
            enable_job_persistence=False,
            max_queue_size=10,
            max_jobs=10,
            job_cleanup_interval_seconds=3600,
            job_retention_seconds=_RETENTION_SECONDS,
            job_max_stuck_time_seconds=3600,
        ),
    )
    clients = []

    def build(runner: str, delay: float = 0.0):
        device = _FakeDevice(delay)
        service = object.__new__(GenerationService)
        service.scheduler = device
        service.logger = SimpleNamespace(
            error=lambda *_: None, info=lambda *_: None, warning=lambda *_: None
        )
        service._job_manager = JobManager()
        service._processManager = SimpleNamespace(
            Event=threading.Event, Namespace=SimpleNamespace
        )
        service._createStartEvent = lambda: asyncio.Event()
        service.process = device.process

        app = FastAPI()
        app.include_router(_router_module(monkeypatch, runner).router)
        app.dependency_overrides[service_resolver] = lambda: service
        app.dependency_overrides[get_api_key] = lambda: None
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        clients.append((client, service))
        return client, device

    yield build
    for client, service in clients:
        service._job_manager._cleanup_task.cancel()


async def _wait_final(client, job_id: str) -> dict:
    for _ in range(200):
        job = (await client.get(f"/v1/jobs/{job_id}")).json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish: {job}")


async def test_image_job_lifecycle(generations_app, tmp_path):
    client, device = generations_app("tt-qwen-image-edit-2511")
    response = await client.post(
        "/v1/generations/qwen/qwen-image-edit-2511",
        json={
            "prompt": "add a hat",
            "images": [_data_uri(1600, 900)],
            "seed": 7,
            "format": "jpeg",
        },
    )
    assert response.status_code == 202, response.text
    job = response.json()
    assert set(job) == {"id", "model", "status", "created_at"}
    assert job["model"] == "qwen/qwen-image-edit-2511"

    job = await _wait_final(client, job["id"])
    assert job["status"] == "completed"
    assert job["expires_at"] == job["completed_at"] + _RETENTION_SECONDS
    (request,) = device.requests
    assert (request._seed, request._prompt) == (7, "add a hat")
    assert request.negative_prompt == " "

    (output,) = job["outputs"]
    assert output == {
        "type": "image",
        "url": f"/v1/jobs/{job['id']}/content/0",
        "content_type": "image/jpeg",
        "seed": 7,
        "prompt": "add a hat",
        "width": 1376,
        "height": 768,
    }
    content = await client.get(output["url"])
    assert content.headers["content-type"] == "image/jpeg"
    assert Image.open(BytesIO(content.content)).size == (1376, 768)
    assert (await client.get(f"/v1/jobs/{job['id']}/content/1")).status_code == 404

    assert (await client.get("/v1/jobs")).json() == [job]
    assert (await client.delete(f"/v1/jobs/{job['id']}")).json() == {
        "id": job["id"],
        "deleted": True,
    }
    assert not (tmp_path / job["id"]).exists()
    assert (await client.get(f"/v1/jobs/{job['id']}")).status_code == 404


async def test_video_job(generations_app):
    client, device = generations_app("tt-wan2.2-i2v")
    response = await client.post(
        "/v1/generations/wan-ai/wan2.2-i2v-a14b",
        json={
            "prompt": "a boat",
            "first_frame": _data_uri(900, 1600),
            "aspect_ratio": "9:16",
            "size": "480p",
        },
    )
    assert response.status_code == 202, response.text
    job = await _wait_final(client, response.json()["id"])
    assert job["status"] == "completed", job
    assert device.requests[0].negative_prompt == WAN_NEGATIVE_PROMPT
    (output,) = job["outputs"]
    assert output["type"] == "video" and output["content_type"] == "video/mp4"
    assert (output["width"], output["height"], output["fps"]) == (480, 832, 16)
    assert output["num_frames"] == 81 and output["duration"] == 81 / 16
    assert output["audio"] is False
    content = await client.get(output["url"])
    assert content.headers["content-type"] == "video/mp4" and content.content


def _sound_video(frames: int = 9) -> VideoAudioResult:
    return VideoAudioResult(
        frames=np.zeros((frames, 64, 64, 3), np.uint8),
        audio=np.zeros((2, 16000 * frames // 25), np.float32),
        sampling_rate=16000,
        fps=25,
    )


@pytest.mark.parametrize("encoded_by_the_pipeline", [False, True])
async def test_video_with_sound_job(generations_app, encoded_by_the_pipeline):
    import av

    client, device = generations_app("tt-ltx-2.3-distilled")
    if encoded_by_the_pipeline:
        # As LTX's generate() does: an MP4 with sound, not faststart.
        def video(request):
            result = _sound_video()
            return VideoManager().export_to_mp4_with_audio(
                result.frames, result.audio, result.sampling_rate, fps=result.fps
            )

        device.video = video
    else:
        device.video = lambda request: _sound_video()
    response = await client.post(
        "/v1/generations/lightricks/ltx-2.3-distilled",
        json={"prompt": "a violinist", "duration": 6},
    )
    assert response.status_code == 202, response.text
    job = await _wait_final(client, response.json()["id"])
    assert job["status"] == "completed", job
    (output,) = job["outputs"]
    assert output["audio"] is True and output["fps"] == 25
    assert output["num_frames"] == 9
    content = await client.get(output["url"])
    with av.open(BytesIO(content.content)) as container:
        assert container.streams.video and container.streams.audio
    # faststart: the index comes before the media data.
    assert content.content.index(b"moov") < content.content.index(b"mdat")


async def test_invalid_media_is_422_before_queueing(generations_app):
    client, device = generations_app("tt-qwen-image-edit-2511")
    not_an_image = "data:image/png;base64," + base64.b64encode(b"nope").decode()
    response = await client.post(
        "/v1/generations/qwen/qwen-image-edit-2511",
        json={"prompt": "x", "images": [_data_uri(), not_an_image]},
    )
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert error["loc"] == ["body", "images", 1]
    assert device.requests == [] and (await client.get("/v1/jobs")).json() == []


async def test_disallowed_url_is_400_and_large_image_413(generations_app, monkeypatch):
    from utils.media_downloader import MediaDownloadPolicyError

    async def refuse(url, **kwargs):
        raise MediaDownloadPolicyError("host is not allowed")

    monkeypatch.setattr("model_services.generation_service.download_media_url", refuse)
    client, _ = generations_app("tt-qwen-image-edit-2511")
    disallowed = await client.post(
        "/v1/generations/qwen/qwen-image-edit-2511",
        json={"prompt": "x", "images": ["https://example.com/a.png"]},
    )
    assert disallowed.status_code == 400
    assert disallowed.json()["detail"] == "host is not allowed"

    monkeypatch.setattr("model_services.generation_service.MEDIA_MAX_BYTES", 10)
    large = await client.post(
        "/v1/generations/qwen/qwen-image-edit-2511",
        json={"prompt": "x", "images": [_data_uri()]},
    )
    assert large.status_code == 413
    assert (await client.get("/v1/jobs")).json() == []


async def test_schema_errors_are_422(generations_app):
    client, _ = generations_app("tt-ideogram-4")
    response = await client.post(
        "/v1/generations/ideogram-ai/ideogram-4",
        json={"structured_prompt": _CAPTION, "size": "8K", "colour": "red"},
    )
    assert response.status_code == 422
    locs = {tuple(error["loc"]) for error in response.json()["detail"]}
    assert {("body", "colour"), ("body", "size")} <= locs
    other = await client.post("/v1/generations/qwen/qwen-image-edit-2511", json={})
    assert other.status_code == 404


async def test_cancel_reaches_the_device_worker(generations_app):
    client, device = generations_app("tt-flux.2-klein-4b", delay=0.5)
    job = (
        await client.post(
            "/v1/generations/black-forest-labs/flux.2-klein-4b",
            json={"prompt": "a cat"},
        )
    ).json()
    while not device.requests:
        await asyncio.sleep(0.01)
    (request,) = device.requests
    assert not request._cancel_event.is_set()
    await client.post(f"/v1/jobs/{job['id']}/cancel")
    # A worker skips a request whose cancel event is set, or stops early.
    assert request._cancel_event.is_set()
    assert (await _wait_final(client, job["id"]))["status"] == "cancelled"


async def test_cancel_and_delete_rules(generations_app):
    client, _ = generations_app("tt-flux.2-klein-4b", delay=0.5)
    job = (
        await client.post(
            "/v1/generations/black-forest-labs/flux.2-klein-4b",
            json={"prompt": "a cat"},
        )
    ).json()

    assert (await client.delete(f"/v1/jobs/{job['id']}")).status_code == 409
    assert (await client.get(f"/v1/jobs/{job['id']}/content/0")).status_code == 409
    cancelled = await client.post(f"/v1/jobs/{job['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] in ("cancelling", "cancelled")
    job = await _wait_final(client, job["id"])
    assert job["status"] == "cancelled" and "outputs" not in job
    again = await client.post(f"/v1/jobs/{job['id']}/cancel")
    assert again.status_code == 409
    assert (await client.post("/v1/jobs/nope/cancel")).status_code == 404
    assert (await client.delete(f"/v1/jobs/{job['id']}")).status_code == 200


async def test_ideogram_gets_the_serialized_caption(generations_app):
    client, device = generations_app("tt-ideogram-4")
    response = await client.post(
        "/v1/generations/ideogram-ai/ideogram-4",
        json={"structured_prompt": _CAPTION, "aspect_ratio": "16:9", "size": "1K"},
    )
    job = await _wait_final(client, response.json()["id"])
    caption = (
        '{"compositional_deconstruction":{"background":"a brick wall",'
        '"elements":[{"type":"text","bbox":[10,20,30,40],"text":"OPEN"}]}}'
    )
    assert device.requests[0]._prompt == caption
    (output,) = job["outputs"]
    assert output["prompt"] == caption
    assert (output["width"], output["height"]) == (1280, 720)


async def test_failed_job_reports_error(generations_app, tmp_path):
    client, device = generations_app("tt-flux.2-klein-4b")
    device.error = RuntimeError("device fell over")
    response = await client.post(
        "/v1/generations/black-forest-labs/flux.2-klein-4b", json={"prompt": "a cat"}
    )
    job = await _wait_final(client, response.json()["id"])
    assert job["status"] == "failed" and "outputs" not in job
    assert job["error"] == {"code": "processing_error", "message": "device fell over"}
    assert job["expires_at"] > job["completed_at"]
    assert (await client.get(f"/v1/jobs/{job['id']}/content/0")).status_code == 409
    assert list(tmp_path.iterdir()) == []


async def test_a_generations_dir_in_any_spelling_works(
    generations_app, monkeypatch, tmp_path
):
    monkeypatch.setenv("TT_GENERATIONS_DIR", f"{tmp_path}//./")
    client, _ = generations_app("tt-flux.2-klein-4b")
    response = await client.post(
        "/v1/generations/black-forest-labs/flux.2-klein-4b", json={"prompt": "a cat"}
    )
    job = await _wait_final(client, response.json()["id"])
    assert job["status"] == "completed", job
    assert (tmp_path / job["id"] / "outputs.json").exists()


async def test_what_a_runner_leaves_in_a_failed_jobs_directory_goes_with_it(
    generations_app, tmp_path
):
    # As LTX's runner writes its video into the job's directory, here before
    # the job fails.
    client, device = generations_app("tt-ltx-2.3-distilled")

    def video(request):
        (tmp_path / request._task_id).mkdir()
        (tmp_path / request._task_id / "ltx.mp4").write_bytes(b"mp4")
        raise RuntimeError("device fell over")

    device.video = video
    response = await client.post(
        "/v1/generations/lightricks/ltx-2.3-distilled", json={"prompt": "a violinist"}
    )
    job = await _wait_final(client, response.json()["id"])
    assert job["status"] == "failed"
    assert (await client.delete(f"/v1/jobs/{job['id']}")).status_code == 200
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("size", "status"), [(12, 422), (20, 422), (9, 202)])
async def test_oversized_images_are_422(generations_app, monkeypatch, size, status):
    # Over the limit (12x12) and over twice it, where PIL itself refuses (20x20).
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    client, _ = generations_app("tt-qwen-image-edit-2511")
    response = await client.post(
        "/v1/generations/qwen/qwen-image-edit-2511",
        json={"prompt": "x", "images": [_data_uri(size, size)]},
    )
    assert response.status_code == status, response.text
    if status == 422:
        assert "more than 100 pixels" in response.json()["detail"][0]["msg"]


def test_generation_job_type_and_manifest_roundtrip(tmp_path):
    from model_services.generation_service import MANIFEST, read_manifest

    (tmp_path / MANIFEST).write_text(json.dumps([{"file": "0.png"}]))
    assert read_manifest(str(tmp_path)) == [{"file": "0.png"}]
    assert JobTypes.GENERATION.value == "generation"


async def test_an_earlier_output_is_an_input_by_its_url(generations_app):
    client, device = generations_app("tt-qwen-image-edit-2511")
    first = (
        await client.post(
            "/v1/generations/qwen/qwen-image-edit-2511",
            json={"prompt": "a hat", "images": [_data_uri(1600, 900)]},
        )
    ).json()
    (output,) = (await _wait_final(client, first["id"]))["outputs"]

    response = await client.post(
        "/v1/generations/qwen/qwen-image-edit-2511",
        json={"prompt": "a scarf", "images": [output["url"]]},
    )
    assert response.status_code == 202, response.text
    assert (await _wait_final(client, response.json()["id"]))["status"] == "completed"
    assert device.requests[-1]._images["images"][0].size == (1376, 768)

    missing = await client.post(
        "/v1/generations/qwen/qwen-image-edit-2511",
        json={"prompt": "x", "images": ["/v1/jobs/nope/content/0"]},
    )
    assert missing.status_code == 422
    assert missing.json()["detail"][0]["loc"] == ["body", "images", 0]


async def test_an_unfinished_jobs_output_is_not_an_input(generations_app):
    client, _ = generations_app("tt-qwen-image-edit-2511", delay=0.5)
    running = (
        await client.post(
            "/v1/generations/qwen/qwen-image-edit-2511",
            json={"prompt": "a hat", "images": [_data_uri(1600, 900)]},
        )
    ).json()
    response = await client.post(
        "/v1/generations/qwen/qwen-image-edit-2511",
        json={"prompt": "a scarf", "images": [f"/v1/jobs/{running['id']}/content/0"]},
    )
    assert response.status_code == 422
    assert "has no completed result" in response.json()["detail"][0]["msg"]


def test_stored_request_digests_data_uris():
    from model_services.generation_service import _request_parameters

    image = _data_uri()
    request = QwenImageEdit2511Request(
        prompt="x", images=[image, "DATA" + image[4:], "https://x/a.png"]
    )
    digest, upper_digest, url = _request_parameters(request)["images"]
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64
    assert upper_digest.startswith("sha256:")
    assert url == "https://x/a.png"


def test_images_are_turned_upright():
    from model_services.generation_service import _decode_image

    # Stored landscape, shown portrait: EXIF orientation 6 is "rotate 90° CW".
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = BytesIO()
    Image.new("RGB", (40, 30)).save(buffer, format="JPEG", exif=exif)
    assert _decode_image(buffer.getvalue()).size == (30, 40)


async def test_lost_outputs_are_404(generations_app, tmp_path):
    client, _ = generations_app("tt-flux.2-klein-4b")
    response = await client.post(
        "/v1/generations/black-forest-labs/flux.2-klein-4b", json={"prompt": "a cat"}
    )
    job = await _wait_final(client, response.json()["id"])
    for path in (tmp_path / job["id"]).iterdir():
        path.unlink()
    (tmp_path / job["id"]).rmdir()
    assert (await client.get(f"/v1/jobs/{job['id']}")).status_code == 404
    assert (await client.get(f"/v1/jobs/{job['id']}/content/0")).status_code == 404
    assert (await client.get("/v1/jobs")).json() == []


async def test_outputs_of_a_job_cancelled_while_writing_go_with_the_job(
    generations_app, monkeypatch, tmp_path
):
    # A running job is cancelled cooperatively: it finishes, becomes
    # `cancelled` without outputs, and its files are deleted with the job.
    from model_services.generation_service import GenerationService

    writing, release = threading.Event(), threading.Event()
    write_outputs = GenerationService._write_outputs

    def slow_write(self, request, results):
        writing.set()
        release.wait(5)
        return write_outputs(self, request, results)

    monkeypatch.setattr(GenerationService, "_write_outputs", slow_write)
    client, _ = generations_app("tt-flux.2-klein-4b")
    job = (
        await client.post(
            "/v1/generations/black-forest-labs/flux.2-klein-4b",
            json={"prompt": "a cat"},
        )
    ).json()
    await asyncio.to_thread(writing.wait, 5)
    await client.post(f"/v1/jobs/{job['id']}/cancel")
    release.set()
    job = await _wait_final(client, job["id"])
    assert job["status"] == "cancelled" and "outputs" not in job
    assert (await client.get(f"/v1/jobs/{job['id']}/content/0")).status_code == 409
    assert (await client.delete(f"/v1/jobs/{job['id']}")).status_code == 200
    assert not (tmp_path / job["id"]).exists()
