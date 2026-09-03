"""Small bounded HTTP client for JSON API requests."""
from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping

import requests


class HttpRequestError(RuntimeError):
    """Raised when an HTTP request cannot produce a JSON response."""


class JsonHttpClient:
    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        timeout: float = 20.0,
        max_attempts: int = 8,
        base_backoff: float = 1.0,
        max_backoff: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.sleep = sleep
        self.random_fn = random_fn

    def get_json(
        self, url: str, params: Mapping[str, object] | None = None
    ) -> object:
        for attempt in range(self.max_attempts):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == self.max_attempts - 1:
                    raise HttpRequestError(
                        f"HTTP request failed for {url}; status unavailable; "
                        f"attempt {attempt + 1}/{self.max_attempts}"
                    ) from exc
                self.sleep(self._backoff(attempt))
                continue

            status = response.status_code
            if 200 <= status < 300:
                try:
                    return response.json()
                except ValueError as exc:
                    raise HttpRequestError(
                        f"invalid JSON from {url}; status {status}; "
                        f"attempt {attempt + 1}/{self.max_attempts}"
                    ) from exc

            retryable = status == 429 or 500 <= status < 600
            if not retryable or attempt == self.max_attempts - 1:
                raise HttpRequestError(
                    f"HTTP {status} for {url}; attempt "
                    f"{attempt + 1}/{self.max_attempts}"
                )

            retry_after = self._retry_after(response)
            self.sleep(retry_after if retry_after is not None else self._backoff(attempt))

        raise AssertionError("unreachable")

    def _backoff(self, attempt: int) -> float:
        return min(self.max_backoff, self.base_backoff * 2**attempt) + self.random_fn()

    def _retry_after(self, response: requests.Response) -> float | None:
        try:
            value = float(response.headers.get("Retry-After", ""))
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None
