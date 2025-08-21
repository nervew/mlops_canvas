from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from app.m00_instalador.service import _default_requirements_path, install_requirements


def test_default_requirements_path_points_to_repo_root():
    path = _default_requirements_path()
    assert path.name == "requirements.txt"
    assert path.exists()


def test_install_requirements_missing_file(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.txt"
    install_requirements(missing)
    captured = capsys.readouterr()
    assert "requirements.txt no encontrado" in captured.out


def test_install_requirements_calls_subprocess(tmp_path, monkeypatch):
    req_file = tmp_path / "req.txt"
    req_file.write_text("pytest")
    fake_result = SimpleNamespace(returncode=0, stdout="ok", stderr="")
    called = {}

    def fake_run(cmd, stdout=None, stderr=None, text=None):
        called["cmd"] = cmd
        return fake_result

    monkeypatch.setattr("app.m00_instalador.service.subprocess.run", fake_run)
    install_requirements(req_file)
    assert called["cmd"] == [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
