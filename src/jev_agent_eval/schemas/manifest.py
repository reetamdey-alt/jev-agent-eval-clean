"""Dataset manifest schema (spec section 14)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CaseFileEntry(BaseModel):
    path: str
    # v2: a file may hold many capabilities (release-v2 canonical store);
    # capability is required only for single-capability v1-style files.
    capability: str | None = None
    count: int | None = None  # advisory; the file itself is authoritative


class DatasetSource(BaseModel):
    name: str
    reference: str
    upstream_repo: str | None = None
    upstream_revision: str | None = None
    transformation_version: str | None = None
    license: str | None = None
    download_timestamp: str | None = None
    source_checksum: str | None = None
    source_sample_ids: list[str] = Field(default_factory=list)


class DatasetManifest(BaseModel):
    name: str
    version: str
    schema_version: str = "1.0"
    license: str = "internal"
    cases: list[CaseFileEntry] = Field(default_factory=list)
    sources: list[DatasetSource] = Field(default_factory=list)
    integrity: dict[str, str] = Field(default_factory=dict)  # path -> sha256
