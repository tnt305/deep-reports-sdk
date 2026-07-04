"""Tiny fixture repo — minimal Python package used in integration tests."""

def add(a: int, b: int) -> int:
    """Return sum of two integers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Return product of two integers."""
    return a * b


def subtract(a: int, b: int) -> int:
    """Return a minus b."""
    return a - b
