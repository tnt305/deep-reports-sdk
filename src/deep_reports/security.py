"""Path traversal guard — enforces allowed_source_roots before any I/O."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence


class PathNotAllowed(Exception):
    """Raised when a source path falls outside allowed_source_roots."""

    pass


def _resolve(path: str | Path) -> Path:
    """Resolve symlinks and normalize. Returns absolute Path."""
    return Path(path).resolve()


def validate_path(path: str | Path, *, allowed_roots: Sequence[str | Path]) -> Path:
    """
    Validate that *path* is contained within one of *allowed_roots*.

    Symlinks are resolved before comparison so a symlink pointing outside
    allowed_roots is rejected even if the symlink path would otherwise match.

    Raises:
        PathNotAllowed: if resolved path is not under any allowed root.
    """
    if not allowed_roots:
        raise ValueError("allowed_roots must be non-empty")

    resolved = _resolve(path)
    roots = [_resolve(r) for r in allowed_roots]

    for root in roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue

    allowed_str = ", ".join(str(r) for r in roots)
    raise PathNotAllowed(
        f"Path {path!r} resolves to {resolved}, which is not under "
        f"any allowed root: {allowed_str}"
    )


def validate_paths(
    paths: Sequence[str | Path], *, allowed_roots: Sequence[str | Path]
) -> list[Path]:
    """Validate a sequence of paths. Returns resolved paths in same order."""
    return [validate_path(p, allowed_roots=allowed_roots) for p in paths]


def get_allowed_roots() -> list[str]:
    """
    Resolve allowed_source_roots from environment.

    Priority:
      1. DR_ALLOWED_SOURCE_ROOTS env var (comma-separated)
      2. "." (current working directory — restrictive default)

    The restrictive default means users must explicitly opt in to broader
    access by setting DR_ALLOWED_SOURCE_ROOTS to e.g. "/home/user/projects".
    """
    env_val = os.getenv("DR_ALLOWED_SOURCE_ROOTS", "").strip()
    if env_val:
        return [r.strip() for r in env_val.split(",") if r.strip()]
    return ["."]
