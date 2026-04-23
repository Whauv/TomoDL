"""Centralized application errors and CLI-safe handling."""

from __future__ import annotations

import functools
import traceback
from typing import Callable, TypeVar


class TomoDLError(Exception):
    """Base project exception type."""


class ConfigurationError(TomoDLError):
    """Raised when configuration is invalid."""


class DataValidationError(TomoDLError):
    """Raised when input data contracts are not satisfied."""


class InferenceError(TomoDLError):
    """Raised when inference fails due to runtime conditions."""


F = TypeVar("F", bound=Callable[..., None])


def cli_entrypoint(func: F) -> F:
    """Wrap CLI entrypoints with centralized error handling."""

    @functools.wraps(func)
    def _wrapped(*args, **kwargs) -> None:
        try:
            func(*args, **kwargs)
        except TomoDLError as exc:
            print(f"[TomoDL Error] {exc}")
            raise SystemExit(2) from exc
        except FileNotFoundError as exc:
            print(f"[File Error] {exc}")
            raise SystemExit(2) from exc
        except Exception as exc:  # noqa: BLE001 - catch-all for CLI stability
            print("[Unhandled Error] Unexpected failure in CLI entrypoint.")
            print(str(exc))
            print(traceback.format_exc())
            raise SystemExit(1) from exc

    return _wrapped  # type: ignore[return-value]
