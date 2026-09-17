from __future__ import annotations

from pathlib import Path

from tests.conftest import write_color_video

from depth_capture.history import JobStore


def test_add_list_get_roundtrip(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.json")
    source = write_color_video(tmp_path / "in.mp4", frames=4)
    output = write_color_video(tmp_path / "out.mp4", frames=4)
    job = store.add(
        title="fight",
        source_path=source,
        output_path=output,
        params={"scale": 0.5, "gamma": 0.7},
        preview_dir=tmp_path / "previews",
    )
    assert job.id
    listed = store.list_jobs()
    assert len(listed) == 1
    assert listed[0].title == "fight"
    fetched = store.get(job.id)
    assert fetched is not None
    assert fetched.params["scale"] == 0.5
    assert fetched.source_exists()
    assert fetched.output_exists()


def test_newest_first(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.json")
    first = store.add(title="a", source_path="s1", output_path="o1")
    second = store.add(title="b", source_path="s2", output_path="o2")
    ids = [job.id for job in store.list_jobs()]
    assert ids[0] == second.id
    assert ids[1] == first.id


def test_delete_removes_files(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.json")
    source = write_color_video(tmp_path / "in.mp4", frames=3)
    output = write_color_video(tmp_path / "out.mp4", frames=3)
    preview = tmp_path / "previews"
    preview.mkdir()
    (preview / "preview_0001.jpg").write_bytes(b"x")
    job = store.add(
        title="clip",
        source_path=source,
        output_path=output,
        preview_dir=preview,
    )
    assert store.delete(job.id) is True
    assert store.get(job.id) is None
    assert store.list_jobs() == []
    assert not source.exists()
    assert not output.exists()
    assert not preview.exists()


def test_list_keeps_jobs_with_missing_files(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.json")
    job = store.add(
        title="gone",
        source_path=tmp_path / "missing_in.mp4",
        output_path=tmp_path / "missing_out.mp4",
    )
    listed = store.list_jobs()
    assert len(listed) == 1
    assert listed[0].id == job.id
    assert listed[0].source_exists() is False
    assert listed[0].output_exists() is False


def test_delete_keeps_shared_source(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.json")
    source = write_color_video(tmp_path / "shared.mp4", frames=3)
    out_a = write_color_video(tmp_path / "a.mp4", frames=3)
    out_b = write_color_video(tmp_path / "b.mp4", frames=3)
    first = store.add(title="a", source_path=source, output_path=out_a)
    store.add(title="b", source_path=source, output_path=out_b)
    store.delete(first.id)
    assert source.exists()
    assert not out_a.exists()
    assert out_b.exists()


def test_delete_unknown_id(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.json")
    assert store.delete("nope") is False


def test_clear_empties_store(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.json")
    store.add(title="a", source_path="s1", output_path="o1")
    store.add(title="b", source_path="s2", output_path="o2")
    store.clear()
    assert store.list_jobs() == []
