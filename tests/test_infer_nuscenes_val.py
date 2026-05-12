from pathlib import Path

import yaml

from scripts.infer_nuscenes_val import resolve_checkpoint_path, resolve_output_dir


def test_resolve_checkpoint_prefers_explicit_path(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    explicit = tmp_path / "custom.pt"
    explicit.write_bytes(b"checkpoint")
    latest = folder / "latest.pt"
    latest.write_bytes(b"latest")

    assert resolve_checkpoint_path(str(folder), str(explicit), None) == explicit.resolve()


def test_resolve_checkpoint_falls_back_to_best_then_latest(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    latest = folder / "latest.pt"
    latest.write_bytes(b"latest")
    best = folder / "best_open_loop.pt"
    best.write_bytes(b"best")

    assert resolve_checkpoint_path(str(folder), None, None) == best.resolve()

    best.unlink()
    assert resolve_checkpoint_path(str(folder), None, None) == latest.resolve()


def test_resolve_checkpoint_uses_relative_resume(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    resume = folder / "checkpoints" / "resume.pt"
    resume.parent.mkdir()
    resume.write_bytes(b"resume")

    assert resolve_checkpoint_path(str(folder), None, "checkpoints/resume.pt") == resume.resolve()


def test_resolve_output_dir_defaults_under_config_folder(tmp_path):
    folder = tmp_path / "run"
    checkpoint = tmp_path / "weights" / "best_open_loop.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")

    assert resolve_output_dir(str(folder), str(checkpoint), None) == (folder / "pure_eval" / "best_open_loop").resolve()
