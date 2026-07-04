"""Unit tests for security.py — path traversal guard."""

from __future__ import annotations

import os

import pytest

from deep_reports.security import (
    PathNotAllowed,
    get_allowed_roots,
    validate_path,
    validate_paths,
)


class TestValidatePath:
    def test_accepts_file_inside_root(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        (root / "src").mkdir()
        target = root / "src" / "main.py"
        target.touch()
        result = validate_path(target, allowed_roots=[str(root)])
        assert result == target.resolve()

    def test_accepts_dir_inside_root(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        result = validate_path(root, allowed_roots=[str(root)])
        assert result == root.resolve()

    def test_accepts_file_at_root_boundary(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        target = root / "file.py"
        target.touch()
        result = validate_path(target, allowed_roots=[str(root)])
        assert result == target.resolve()

    def test_rejects_path_outside_root(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        target = tmp_path / "other" / "file.py"
        target.parent.mkdir()
        target.touch()
        with pytest.raises(PathNotAllowed) as exc:
            validate_path(target, allowed_roots=[str(root)])
        assert "not under any allowed root" in str(exc.value)

    def test_rejects_absolute_path_outside_root(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        target = tmp_path / "outside.txt"
        target.touch()
        with pytest.raises(PathNotAllowed):
            validate_path(str(target), allowed_roots=[str(root)])

    def test_rejects_etc_passwd(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        with pytest.raises(PathNotAllowed):
            validate_path("/etc/passwd", allowed_roots=[str(root)])

    def test_rejects_toplevel_traversal(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        with pytest.raises(PathNotAllowed):
            validate_path("../../../etc/passwd", allowed_roots=[str(root)])

    def test_rejects_empty_allowed_roots(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_path("/etc/passwd", allowed_roots=[])

    def test_multiple_allowed_roots(self, tmp_path):
        root1 = tmp_path / "proj1"
        root2 = tmp_path / "proj2"
        root1.mkdir()
        root2.mkdir()
        target = root2 / "file.py"
        target.touch()
        result = validate_path(target, allowed_roots=[str(root1), str(root2)])
        assert result == target.resolve()

    def test_symlink_to_outside_rejected(self, tmp_path):
        """Symlink pointing outside allowed_roots must be rejected."""
        root = tmp_path / "project"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").touch()

        link = root / "link_to_secret"
        link.symlink_to(outside / "secret.txt")

        with pytest.raises(PathNotAllowed):
            validate_path(link, allowed_roots=[str(root)])


class TestValidatePaths:
    def test_validates_multiple_paths(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        f1 = root / "a.py"
        f2 = root / "b.py"
        f1.touch()
        f2.touch()
        results = validate_paths([f1, f2], allowed_roots=[str(root)])
        assert results == [f1.resolve(), f2.resolve()]

    def test_raises_on_first_invalid(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        good = root / "good.py"
        good.touch()
        bad = tmp_path / "bad.py"
        with pytest.raises(PathNotAllowed):
            validate_paths([good, bad], allowed_roots=[str(root)])


class TestGetAllowedRoots:
    def test_defaults_to_cwd(self):
        orig = os.environ.pop("DR_ALLOWED_SOURCE_ROOTS", None)
        try:
            roots = get_allowed_roots()
            assert roots == ["."]
        finally:
            if orig is not None:
                os.environ["DR_ALLOWED_SOURCE_ROOTS"] = orig

    def test_reads_env_var(self, tmp_path):
        # Direct test of env parsing
        os.environ["DR_ALLOWED_SOURCE_ROOTS"] = str(tmp_path)
        try:
            result = get_allowed_roots()
            assert result == [str(tmp_path)]
        finally:
            del os.environ["DR_ALLOWED_SOURCE_ROOTS"]

    def test_parses_comma_separated(self, tmp_path):
        p1 = tmp_path / "a"
        p2 = tmp_path / "b"
        p1.mkdir(exist_ok=True)
        p2.mkdir(exist_ok=True)
        os.environ["DR_ALLOWED_SOURCE_ROOTS"] = f"{p1},{p2}"
        try:
            result = get_allowed_roots()
            assert result == [str(p1), str(p2)]
        finally:
            del os.environ["DR_ALLOWED_SOURCE_ROOTS"]
