from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class JobRecord:
    id: str
    title: str
    created_at: str
    source_path: str
    output_path: str
    params: dict[str, Any] = field(default_factory=dict)
    preview_dir: str | None = None

    def source_exists(self) -> bool:
        return Path(self.source_path).is_file()

    def output_exists(self) -> bool:
        return Path(self.output_path).is_file()

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "params": dict(self.params),
            "source_exists": self.source_exists(),
            "output_exists": self.output_exists(),
        }


class JobStore:
    """JSON-backed processing history under output/jobs.json."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def list_jobs(self) -> list[JobRecord]:
        jobs = self._load()
        jobs.reverse()
        return jobs

    def get(self, job_id: str) -> JobRecord | None:
        for job in self._load():
            if job.id == job_id:
                return job
        return None

    def add(
        self,
        *,
        title: str,
        source_path: str | Path,
        output_path: str | Path,
        params: dict[str, Any] | None = None,
        preview_dir: str | Path | None = None,
    ) -> JobRecord:
        job = JobRecord(
            id=uuid.uuid4().hex[:12],
            title=title.strip() or "untitled",
            created_at=_now_iso(),
            source_path=str(Path(source_path)),
            output_path=str(Path(output_path)),
            params=dict(params or {}),
            preview_dir=str(preview_dir) if preview_dir else None,
        )
        jobs = self._load()
        jobs.append(job)
        self._save(jobs)
        return job

    def delete(self, job_id: str, *, delete_files: bool = True) -> bool:
        jobs = self._load()
        match = next((job for job in jobs if job.id == job_id), None)
        if match is None:
            return False
        remaining = [job for job in jobs if job.id != job_id]
        self._save(remaining)
        if delete_files:
            _remove_path(match.output_path)
            if match.preview_dir:
                preview = Path(match.preview_dir)
                if preview.is_dir():
                    shutil.rmtree(preview, ignore_errors=True)
            source_still_used = any(job.source_path == match.source_path for job in remaining)
            if not source_still_used:
                _remove_path(match.source_path)
        return True

    def clear(self) -> None:
        self._save([])

    def _load(self) -> list[JobRecord]:
        if not self.path.is_file():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(raw, list):
            return []
        jobs: list[JobRecord] = []
        for item in raw:
            if not isinstance(item, dict) or "id" not in item:
                continue
            jobs.append(
                JobRecord(
                    id=str(item["id"]),
                    title=str(item.get("title") or "untitled"),
                    created_at=str(item.get("created_at") or ""),
                    source_path=str(item.get("source_path") or ""),
                    output_path=str(item.get("output_path") or ""),
                    params=dict(item.get("params") or {}),
                    preview_dir=item.get("preview_dir"),
                )
            )
        return jobs

    def _save(self, jobs: list[JobRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(job) for job in jobs]
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def _remove_path(path_value: str) -> None:
    path = Path(path_value)
    if path.is_file():
        path.unlink(missing_ok=True)
