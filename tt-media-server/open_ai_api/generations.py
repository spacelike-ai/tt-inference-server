# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The generations API: one submit endpoint for the model this deployment
serves, and the jobs it creates under /v1/jobs. Design:
docs/api/generation/."""

from typing import Annotated, Optional

from config.constants import JobTypes
from config.settings import settings
from domain.generation_job import Deleted, Job
from domain.generation_requests import GENERATION_REQUESTS
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Security
from fastapi.responses import FileResponse
from model_services.generation_service import GenerationService, read_manifest
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

router = APIRouter(dependencies=[Security(get_api_key)])

_NOT_FOUND = "Job not found"
_OUTPUTS_GONE = "Job outputs are gone"


def _job_or_404(service: GenerationService, job_id: str) -> dict:
    # The job database can hold other services' jobs, restored from a store
    # this deployment shares with them.
    metadata = service.get_job_metadata(job_id)
    if metadata is None or metadata["job_type"] != JobTypes.GENERATION.value:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return metadata


def _outputs(service: GenerationService, job_id: str) -> Optional[list[dict]]:
    """A completed job's outputs, with their urls; None when its files are
    gone, e.g. for a job restored from the database after they were lost."""
    result_path = service.get_job_result_path(job_id)
    if result_path is None:
        return None
    try:
        outputs = read_manifest(result_path)
    except FileNotFoundError:
        return None
    return [
        {
            **{k: v for k, v in output.items() if k != "file"},
            "url": f"/v1/jobs/{job_id}/content/{index}",
        }
        for index, output in enumerate(outputs)
    ]


def _job(service: GenerationService, metadata: dict) -> Optional[Job]:
    """The job as the API shows it; None for a completed job whose outputs are
    gone."""
    outputs = None
    if metadata["status"] == "completed":
        outputs = _outputs(service, metadata["id"])
        if outputs is None:
            return None
    return Job.from_metadata(metadata, settings.job_retention_seconds, outputs)


def _add_submit_route(request_model) -> None:
    @router.post(
        f"/v1/generations/{request_model.MODEL_ID}",
        status_code=202,
        response_model=Job,
        tags=[request_model.MODEL_ID],
        summary=f"Submit a {request_model.MODEL_ID} job",
    )
    async def submit(
        request: Annotated[request_model, Body()],
        service: GenerationService = Depends(service_resolver),
    ) -> Job:
        return Job.from_metadata(
            await service.submit(request), settings.job_retention_seconds
        )


if settings.model_runner in GENERATION_REQUESTS:
    _add_submit_route(GENERATION_REQUESTS[settings.model_runner])


@router.get(
    "/v1/jobs",
    response_model=list[Job],
    tags=["Jobs"],
)
def list_jobs(service: GenerationService = Depends(service_resolver)) -> list[Job]:
    """Every job this deployment still holds, any status."""
    jobs = (
        _job(service, metadata)
        for metadata in service.get_all_jobs_metadata(JobTypes.GENERATION)
    )
    return [job for job in jobs if job is not None]


@router.get(
    "/v1/jobs/{job_id}",
    response_model=Job,
    tags=["Jobs"],
)
def get_job(job_id: str, service: GenerationService = Depends(service_resolver)) -> Job:
    """Poll until the status is final (`completed`, `failed`, `cancelled`). A
    completed job carries its `outputs`, so no second call is needed."""
    job = _job(service, _job_or_404(service, job_id))
    if job is None:
        raise HTTPException(status_code=404, detail=_OUTPUTS_GONE)
    return job


@router.delete("/v1/jobs/{job_id}", response_model=Deleted, tags=["Jobs"])
async def delete_job(
    job_id: str, service: GenerationService = Depends(service_resolver)
) -> Deleted:
    """Only a job in a final state can be deleted; cancel an active one first."""
    _job_or_404(service, job_id)
    try:
        deleted = await service.delete_job(job_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    return Deleted(id=job_id)


@router.get("/v1/jobs/{job_id}/content/{index}", tags=["Jobs"])
def get_job_content(
    job_id: str,
    index: Annotated[int, Path(ge=0)],
    service: GenerationService = Depends(service_resolver),
) -> FileResponse:
    """The file behind `outputs[index].url`. Videos are MP4: H.264 (High
    profile, yuv420p), faststart, with an AAC audio track if the output's
    `audio` is true."""
    metadata = _job_or_404(service, job_id)
    result_path = service.get_job_result_path(job_id)
    if metadata["status"] != "completed" or result_path is None:
        raise HTTPException(
            status_code=409, detail=f"Job is {metadata['status']}, not completed"
        )
    try:
        outputs = read_manifest(result_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=_OUTPUTS_GONE)
    if index >= len(outputs):
        raise HTTPException(status_code=404, detail=f"Job has {len(outputs)} outputs")
    output = outputs[index]
    return FileResponse(
        f"{result_path}/{output['file']}", media_type=output["content_type"]
    )


@router.post(
    "/v1/jobs/{job_id}/cancel",
    response_model=Job,
    tags=["Jobs"],
)
def cancel_job(
    job_id: str, service: GenerationService = Depends(service_resolver)
) -> Job:
    """A queued job is cancelled at once (`cancelled`). A running one moves to
    `cancelling`, then to `cancelled` once the pipeline stops."""
    _job_or_404(service, job_id)
    metadata = service.cancel_job(job_id)
    if metadata is None:
        # The job manager declines unknown and already-final jobs alike.
        status = _job_or_404(service, job_id)["status"]
        raise HTTPException(status_code=409, detail=f"Job is already {status}")
    return Job.from_metadata(metadata, settings.job_retention_seconds)
