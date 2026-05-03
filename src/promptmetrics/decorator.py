from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from promptmetrics.models import Trace
from promptmetrics.storage import Storage

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Process-wide storage handle. Created lazily so importing the package
# never touches the filesystem.
_storage: Storage | None = None
_storage_lock = threading.Lock()


def _get_storage() -> Storage:
    global _storage
    if _storage is None:
        with _storage_lock:
            if _storage is None:
                _storage = Storage()
    return _storage


def set_storage(storage: Storage) -> None:
    """Override the default storage. Mostly used by tests."""
    global _storage
    with _storage_lock:
        _storage = storage


def _coerce_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    if len(args) == 1 and not kwargs:
        return str(args[0])
    if not args and len(kwargs) == 1:
        return str(next(iter(kwargs.values())))
    return repr({"args": args, "kwargs": kwargs})


def _extract_token_counts(
    result: Any,
    extractor: Callable[[Any], tuple[int, int]] | None,
    raise_on_error: bool,
) -> tuple[int, int]:
    if extractor is not None:
        try:
            return extractor(result)
        except Exception:
            if raise_on_error:
                raise
            logger.exception("token extractor raised; recording zeros")
            return 0, 0
    usage = getattr(result, "usage", None)
    if usage is not None:
        in_tok = (
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", None)
            or 0
        )
        out_tok = (
            getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", None)
            or 0
        )
        return int(in_tok), int(out_tok)
    return 0, 0


def _extract_output(
    result: Any,
    extractor: Callable[[Any], str] | None,
    raise_on_error: bool,
) -> str:
    if extractor is not None:
        try:
            return str(extractor(result))
        except Exception:
            if raise_on_error:
                raise
            logger.exception("output extractor raised; recording empty string")
            return ""
    return str(result) if result is not None else ""


def track(
    prompt_id: str,
    *,
    model: str | None = None,
    extract_output: Callable[[Any], str] | None = None,
    extract_tokens: Callable[[Any], tuple[int, int]] | None = None,
    raise_on_error: bool = False,
) -> Callable[[F], F]:
    """Capture latency, tokens, input, and output for the wrapped call.

    By default, extractor and storage failures are logged and swallowed so
    observability never breaks the host application. Pass
    ``raise_on_error=True`` to surface those failures — useful in CI / tests
    where silent metric corruption is worse than a loud crash.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000.0

            try:
                in_tok, out_tok = _extract_token_counts(
                    result, extract_tokens, raise_on_error
                )
                trace = Trace(
                    prompt_id=prompt_id,
                    input=_coerce_input(args, kwargs),
                    output=_extract_output(result, extract_output, raise_on_error),
                    latency_ms=latency_ms,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    model=model,
                )
                _get_storage().insert_trace(trace)
            except Exception:
                if raise_on_error:
                    raise
                logger.exception("promptmetrics: failed to record trace")
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
