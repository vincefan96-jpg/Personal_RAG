import time
import functools
import logging
from typing import Callable, TypeVar

logger = logging.getLogger("rag")

T = TypeVar("T")


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Exponential backoff retry decorator for API calls."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = base_delay * (backoff ** attempt)
                        logger.warning(
                            "%s 失败 (第 %d/%d 次), %0.1fs 后重试: %s",
                            func.__name__, attempt + 1, max_retries, delay, e,
                        )
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


_singleton_registry: dict = {}


def singleton(key: str):
    """Decorator: cache result of a zero-arg factory under `key`.

    Usage:
        @singleton("embeddings")
        def get_embeddings():
            return DashScopeEmbeddings(...)
    """

    def decorator(factory: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(factory)
        def wrapper(*args, **kwargs):
            if key not in _singleton_registry:
                _singleton_registry[key] = factory(*args, **kwargs)
            return _singleton_registry[key]

        return wrapper

    return decorator


def reset_singletons(*keys: str):
    """Clear specific singletons, or all if no keys given."""
    if not keys:
        _singleton_registry.clear()
    else:
        for k in keys:
            _singleton_registry.pop(k, None)
