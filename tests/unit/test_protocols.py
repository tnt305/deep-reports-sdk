"""Import smoke tests — verify all public interfaces are importable."""

from __future__ import annotations

import deep_reports


def test_import_deep_reports():
    """deep_reports package imports without error."""
    assert deep_reports.__version__ == "0.1.0"


def test_import_security():
    """security module is importable."""
    from deep_reports import security

    assert hasattr(security, "PathNotAllowed")
    assert hasattr(security, "validate_path")
    assert hasattr(security, "validate_paths")
    assert hasattr(security, "get_allowed_roots")


def test_import_cli():
    """CLI module is importable."""
    from deep_reports import cli

    assert hasattr(cli, "cli")
