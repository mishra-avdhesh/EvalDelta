"""HTTP provider for a candidate served behind a JSON endpoint.

Intended for free, local or self-hosted endpoints (a local model server, a free-tier Space).
No paid API is required or assumed. Authentication is optional: a bearer token is read from an
environment variable named in the config, and is never written to reports or logs.

Request:  POST <url>  body = {"sample_id": ..., "input": <item[input_field]>}
Response: JSON; the output is ``response[output_field]``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from evaldelta.providers.base import EvalOutcome

RETRYABLE = {408, 425, 429, 500, 502, 503, 504}


class HTTPProvider:
    name = "http"

    def __init__(
        self,
        url: str,
        scorer: Callable[[Any, Mapping[str, Any]], float],
        *,
        input_field: str = "input",
        output_field: str = "output",
        token_env: str | None = None,
        timeout_s: float = 30.0,
        retries: int = 3,
        backoff_s: float = 1.0,
        allowed_hosts: Sequence[str] | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("HTTP provider URL must be http(s)")
        if allowed_hosts is not None and parsed.hostname not in set(allowed_hosts):
            raise ValueError(f"host {parsed.hostname!r} not in allowed_hosts")
        self.url = url
        self.scorer = scorer
        self.input_field = input_field
        self.output_field = output_field
        self._token_env = token_env
        self.timeout_s = timeout_s
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.retries = retries
        self.backoff_s = backoff_s
        self._open = opener or urllib.request.urlopen

    @classmethod
    def from_spec(
        cls, spec: Mapping[str, Any], scorer: Callable[[Any, Mapping[str, Any]], float]
    ) -> HTTPProvider:
        return cls(
            spec["url"],
            scorer,
            input_field=spec.get("input_field", "input"),
            output_field=spec.get("output_field", "output"),
            token_env=spec.get("token_env"),
            timeout_s=float(spec.get("timeout_s", 30)),
            retries=int(spec.get("retries", 3)),
            backoff_s=float(spec.get("backoff_s", 1.0)),
            allowed_hosts=spec.get("allowed_hosts"),
        )

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._token_env:
            tok = os.environ.get(self._token_env)
            if tok:
                h["Authorization"] = f"Bearer {tok}"
        return h

    def evaluate(self, items: Sequence[Mapping[str, Any]]) -> list[EvalOutcome]:
        return [self._one(it) for it in items]

    def evaluate_budgeted(
        self, item: Mapping[str, Any], before_attempt: Callable[[], bool]
    ) -> EvalOutcome:
        return self._one(item, before_attempt)

    def _one(
        self, item: Mapping[str, Any], before_attempt: Callable[[], bool] | None = None
    ) -> EvalOutcome:
        sid = str(item["sample_id"])
        body = json.dumps(
            {"sample_id": sid, "input": item.get(self.input_field)}, default=str
        ).encode()
        err: str | None = None
        for attempt in range(1, self.retries + 2):
            if before_attempt is not None and not before_attempt():
                return EvalOutcome(sid, None, error="budget_exhausted", attempts=attempt - 1)
            req = urllib.request.Request(
                self.url, data=body, headers=self._headers(), method="POST"
            )
            try:
                with self._open(req, timeout=self.timeout_s) as resp:
                    payload = json.loads(resp.read().decode())
                output = payload[self.output_field]
            except urllib.error.HTTPError as exc:
                err = f"HTTP {exc.code}"
                if exc.code not in RETRYABLE:
                    break
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
                err = type(exc).__name__
            else:
                loss = float(self.scorer(output, item))
                if not 0.0 <= loss <= 1.0:
                    raise ValueError(f"scorer returned {loss} outside [0, 1]")
                return EvalOutcome(sid, loss, output=output, attempts=attempt)
            if attempt <= self.retries and self.backoff_s:
                time.sleep(self.backoff_s * 2 ** (attempt - 1))
        return EvalOutcome(sid, None, error=err, attempts=attempt)
