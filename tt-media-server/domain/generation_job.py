# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class JobError(BaseModel):
    code: str = Field(
        description=(
            "Why the job failed: `processing_error`, `stale_job` or "
            "`server_restart`. More codes may be added; treat an unknown one "
            "like `processing_error`."
        )
    )
    message: str = Field(description="Human-readable.")


class _OutputBase(BaseModel):
    url: str = Field(
        description=(
            "Where to download the file: a path relative to this server's base "
            "URL, fetched with the same API key. Clients use an absolute URL as "
            "given and resolve a relative one against the base URL."
        )
    )
    content_type: str
    seed: int = Field(
        description="The job's seed (the one chosen at random if the request had none)."
    )
    prompt: str = Field(
        description=(
            "The prompt as the model received it. For Ideogram 4, the "
            "`structured_prompt` caption as compact JSON, in the model's key order."
        )
    )
    width: int
    height: int


class ImageOutput(_OutputBase):
    type: Literal["image"] = "image"


class VideoOutput(_OutputBase):
    type: Literal["video"] = "video"
    fps: int
    num_frames: int
    duration: float = Field(description="Seconds.")


Output = Annotated[Union[ImageOutput, VideoOutput], Field(discriminator="type")]


class Job(BaseModel):
    id: str
    model: str = Field(description="The `{org}/{model}` it was submitted to.")
    status: Literal[
        "queued", "in_progress", "cancelling", "completed", "failed", "cancelled"
    ] = Field(description="Final states are `completed`, `failed` and `cancelled`.")
    created_at: int = Field(description="Unix seconds.")
    completed_at: Optional[int] = Field(
        default=None, description="Unix seconds, once the status is final."
    )
    expires_at: Optional[int] = Field(
        default=None,
        description=(
            "Unix seconds when the job and its output files are deleted, once the "
            "status is final: `completed_at` plus the server's retention period. "
            "After it, the job returns 404."
        ),
    )
    error: Optional[JobError] = Field(default=None, description="Set when `failed`.")
    outputs: Optional[list[Output]] = Field(
        default=None,
        min_length=1,
        description="Set when `completed`: one per image (`count` of them) or one video.",
    )

    @classmethod
    def from_metadata(
        cls,
        metadata: dict,
        retention_seconds: int,
        outputs: Optional[list[dict]] = None,
    ) -> "Job":
        """From the job manager's public job dict."""
        completed_at = metadata.get("completed_at")
        return cls(
            id=metadata["id"],
            model=metadata["model"],
            status=metadata["status"],
            created_at=metadata["created_at"],
            completed_at=completed_at,
            expires_at=completed_at + retention_seconds if completed_at else None,
            error=metadata.get("error"),
            outputs=outputs,
        )


class Deleted(BaseModel):
    id: str
    deleted: Literal[True] = True
