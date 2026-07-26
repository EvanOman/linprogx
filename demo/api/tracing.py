"""Fail-open OpenTelemetry tracing for the Modal-hosted linprogx demo API.

The public site already emits browser spans (Grafana Faro) and Cloudflare Pages
Function spans, and the ``/api/lp`` proxy forwards a W3C ``traceparent`` to this
origin. This module continues that trace inside Modal so the browser -> edge ->
Modal waterfall is one trace instead of three disconnected ones.

Deliberately dependency-free (stdlib only), for three reasons:

1. This repo's contract keeps the runtime dependency-light, and the demo image
   is intentionally tiny. The OpenTelemetry SDK plus its FastAPI instrumentation
   is several hundred milliseconds of import time baked into every Modal cold
   start -- the exact number this instrumentation exists to measure.
2. The Pages Function exporter (``functions/_observability.ts`` in the site repo)
   is already a hand-rolled OTLP/HTTP JSON client. Matching its payload shape,
   env var names, and privacy rules keeps one reviewable contract instead of two.
3. It keeps this repo's dependency set, lockfile, and package release-age policy
   untouched.

The wire format is standard OTLP/HTTP with a JSON payload and standard OTel
semantic conventions, so Grafana Cloud (or any OTLP receiver) ingests it with no
special handling.

Privacy: this module records bounded, enumerated values only -- never node or
edge names, solver request/response bodies, headers, IPs, or exception messages. See ``docs/observability.md`` in the site repo for the shared contract.

Fail-open is a hard requirement: no telemetry path may raise into a user request.
Every export failure is swallowed and rate-limit-logged.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import math
import os
import queue
import re
import secrets
import socket
import threading
import time
import urllib.request
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

AttributeValue = str | bool | int | float

# --- OTLP enums (protobuf numeric values, JSON mapping) ----------------------

KIND_INTERNAL = 1
KIND_SERVER = 2
KIND_CLIENT = 3

STATUS_UNSET = 0
STATUS_OK = 1
STATUS_ERROR = 2

# Losing the client mid-request unwinds the ASGI app with one of these. Neither
# is a server fault, so they are recorded separately from a real failure.
_CANCELLED = (asyncio.CancelledError, GeneratorExit)

# --- Bounds ------------------------------------------------------------------

_MAX_ATTR_CHARS = 128
_MAX_ATTRS = 32
_MAX_QUEUED_SPANS = 2048
_MAX_BATCH = 128
_SCHEDULE_DELAY_S = 0.25
_EXPORT_TIMEOUT_S = 5.0
_FAILURE_LOG_INTERVAL_S = 60.0

# A container that has been idle for longer than this cannot plausibly be
# reporting a real init duration: Modal memory-snapshot restores resume a process
# whose monotonic clock predates the snapshot, so the measured "import to first
# request" gap becomes meaningless. Past the cap we still report the cold start,
# just without a duration.
_MAX_PLAUSIBLE_INIT_S = 120.0

_HEX_2 = re.compile(r"^[0-9a-f]{2}$")
_HEX_16 = re.compile(r"^[0-9a-f]{16}$")
_HEX_32 = re.compile(r"^[0-9a-f]{32}$")
# Accept only random-looking UUID/trace-id shapes so a caller cannot smuggle
# arbitrary text into telemetry through an otherwise "safe" header.
_CORRELATION_ID = re.compile(
    r"^(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.IGNORECASE,
)

# --- Process identity and cold-start bookkeeping -----------------------------

_instance_lock = threading.Lock()
_instance_key: tuple[int, str] | None = None
_instance_value = ""


def instance_id() -> str:
    """Return an id unique to this restored container process.

    This is deliberately lazy. Modal snapshots module globals, so generating an
    id at import time would clone the same value into every container restored
    from that snapshot. The runtime pid/hostname pair changes after a restore;
    when it does, mint a fresh opaque id.
    """
    global _instance_key, _instance_value
    try:
        hostname = socket.gethostname()
    except OSError:
        # Container identity is useful telemetry, never a serving dependency.
        hostname = ""
    key = (os.getpid(), hostname)
    with _instance_lock:
        if key != _instance_key:
            _instance_key = key
            try:
                _instance_value = secrets.token_hex(8)
            except (OSError, RuntimeError):
                # Python salts hash() per process, keeping this opaque while
                # preserving fail-open behavior if the OS CSPRNG is unavailable.
                _instance_value = f"{hash(key) & ((1 << 64) - 1):016x}"
        return _instance_value


def _process_start_monotonic() -> float:
    """Return process start on the monotonic clock when Linux exposes it.

    Capturing the clock only when this module imports misses imports that ran
    before it. Modal runs Linux, where ``/proc/self/stat`` supplies process start
    in clock ticks since boot -- the same origin as ``monotonic()``. The fallback
    keeps local non-Linux development fail-open.
    """
    now = time.monotonic()
    try:
        with open("/proc/self/stat", encoding="utf-8") as stat_file:
            stat = stat_file.read()
        # The second field is parenthesized and may contain spaces, so split only
        # after its closing parenthesis. Field 22 is then index 19 in the tail.
        fields = stat.rpartition(")")[2].split()
        ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
        if ticks_per_second <= 0:
            return now
        started = int(fields[19]) / ticks_per_second
        return started if 0.0 <= started <= now else now
    except (OSError, ValueError, IndexError):
        return now


_init_monotonic = _process_start_monotonic()
_cold_start_lock = threading.Lock()
_served_a_request = False
_request_count = 0


def request_container_state() -> tuple[bool, str, float | None]:
    """Claim the cold-start marker for this process.

    Returns ``(is_cold, reuse_bucket, init_seconds)``. The reuse bucket is
    deliberately bounded (``first``, ``2-5``, ``6-20``, ``21+``) rather than
    exporting an ever-growing request count as a high-cardinality attribute.
    ``init_seconds`` is ``None`` when the measured value is not plausible.
    """
    global _request_count, _served_a_request
    with _cold_start_lock:
        _request_count += 1
        count = _request_count
        is_cold = not _served_a_request
        _served_a_request = True
        elapsed = time.monotonic() - _init_monotonic if is_cold else None
    bucket = "first" if count == 1 else "2-5" if count <= 5 else "6-20" if count <= 20 else "21+"
    init_seconds = (
        elapsed if elapsed is not None and 0.0 <= elapsed <= _MAX_PLAUSIBLE_INIT_S else None
    )
    return is_cold, bucket, init_seconds


def reset_cold_start_for_tests() -> None:
    """Restore the freshly-imported-process state. Test-only.

    Re-anchors the init clock as well as the marker: a long test session would
    otherwise leave an implausible (>120 s) init window and suppress the
    duration, making cold-start assertions depend on how long the suite has
    been running.
    """
    global _request_count, _served_a_request, _init_monotonic
    with _cold_start_lock:
        _served_a_request = False
        _request_count = 0
        _init_monotonic = time.monotonic()


# --- W3C trace context -------------------------------------------------------


@dataclass(frozen=True)
class SpanContext:
    """The identity half of a span: what gets propagated, not what gets recorded."""

    trace_id: str
    span_id: str
    trace_flags: str = "01"

    @property
    def sampled(self) -> bool:
        try:
            return bool(int(self.trace_flags, 16) & 0x01)
        except ValueError:  # pragma: no cover -- constructor inputs are validated
            return False

    def traceparent(self) -> str:
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags}"


def parse_traceparent(value: str | None) -> SpanContext | None:
    """Parse a W3C ``traceparent``, returning ``None`` for anything invalid.

    Mirrors the edge parser: an unparseable, malformed, or all-zero header is not
    an error -- it simply means this request starts a new trace.
    """
    if not value or len(value) > 256:
        return None
    parts = value.strip().split("-")
    if len(parts) < 4:
        return None

    version, trace_id, span_id, flags = (part.lower() for part in parts[:4])
    future_fields = parts[4:]

    if (
        not _HEX_2.match(version)
        or version == "ff"
        or not _HEX_32.match(trace_id)
        or not trace_id.strip("0")
        or not _HEX_16.match(span_id)
        or not span_id.strip("0")
        or not _HEX_2.match(flags)
    ):
        return None

    # Version 00 has exactly four fields. Later versions may append fields, but
    # never empty ones.
    if (version == "00" and future_fields) or any(not field for field in future_fields):
        return None

    return SpanContext(trace_id=trace_id, span_id=span_id, trace_flags=flags)


# --- OTLP attribute encoding -------------------------------------------------


def _attribute(key: str, value: AttributeValue) -> dict[str, Any] | None:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        # NaN/Inf are not representable in JSON; dropping the attribute is better
        # than emitting a payload the collector rejects.
        if not math.isfinite(value):
            return None
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)[:_MAX_ATTR_CHARS]}}


def _attributes(values: Mapping[str, AttributeValue]) -> list[dict[str, Any]]:
    encoded = []
    for key, value in list(values.items())[:_MAX_ATTRS]:
        attribute = _attribute(key, value)
        if attribute is not None:
            encoded.append(attribute)
    return encoded


# --- Exporter ----------------------------------------------------------------


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _traces_endpoint() -> str:
    """Resolve the OTLP traces URL, or "" when tracing is not configured.

    Read lazily on every export rather than cached at import: Modal builds the
    memory snapshot before secrets are necessarily present in the environment, so
    an import-time read could permanently disable a correctly configured deploy.
    """
    if _env("OTEL_SDK_DISABLED").lower() == "true":
        return ""
    explicit = _env("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if explicit:
        url = explicit
    else:
        base = _env("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not base:
            return ""
        url = f"{base.rstrip('/')}/v1/traces"
    # Refuse anything that is not a real HTTP(S) receiver: a file:// or similar
    # endpoint in the environment must not turn into a local-file write.
    return url if url.startswith(("http://", "https://")) else ""


class SpanExporter:
    """Bounded, non-blocking OTLP/HTTP JSON span exporter.

    Spans are handed to a queue and written by a single daemon thread, so no user
    request ever waits on the collector. A full queue drops spans; a failing
    collector is counted and logged at most once a minute, never raised.
    """

    def __init__(self, service_name: str) -> None:
        self.service_name = service_name
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=_MAX_QUEUED_SPANS)
        self._worker: threading.Thread | None = None
        self._worker_pid: int | None = None
        self._worker_lock = threading.Lock()
        self._pending = 0
        self._pending_condition = threading.Condition()
        self._failures = 0
        self._last_failure_log = 0.0
        self._failure_lock = threading.Lock()

    # -- public API --

    @property
    def enabled(self) -> bool:
        return bool(_traces_endpoint())

    def submit(self, span: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._ensure_worker()
        dropped = False
        # Count and enqueue under one lock so the worker cannot finish a span
        # before flush() knows that it is pending.
        with self._pending_condition:
            try:
                self._queue.put_nowait(span)
            except queue.Full:
                dropped = True
            else:
                self._pending += 1
        if dropped:
            self._note_failure()

    def flush(self, timeout: float = 3.0) -> None:
        """Best-effort drain, used at process exit. Never raises."""
        deadline = time.monotonic() + timeout
        with self._pending_condition:
            while self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._pending_condition.wait(remaining)

    # -- worker lifecycle --

    def _ensure_worker(self) -> None:
        """Start the export thread on first use, and after a fork or snapshot restore.

        Starting lazily (rather than at import) keeps the thread out of Modal's
        memory snapshot entirely; the pid/liveness check covers a restore or fork
        that left a dead thread object behind.
        """
        worker = self._worker
        if worker is not None and worker.is_alive() and self._worker_pid == os.getpid():
            return
        with self._worker_lock:
            worker = self._worker
            if worker is not None and worker.is_alive() and self._worker_pid == os.getpid():
                return
            self._worker_pid = os.getpid()
            self._worker = threading.Thread(
                target=self._run,
                name="otlp-span-exporter",
                daemon=True,
            )
            self._worker.start()

    def _run(self) -> None:
        """Drain-and-send forever. Daemon: it ends when the process does."""
        while True:
            batch = self._drain()
            if batch:
                try:
                    self._send(batch)
                finally:
                    with self._pending_condition:
                        self._pending -= len(batch)
                        self._pending_condition.notify_all()

    def _drain(self) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        try:
            batch.append(self._queue.get(timeout=_SCHEDULE_DELAY_S))
        except queue.Empty:
            return batch
        while len(batch) < _MAX_BATCH:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    # -- transport --

    def _resource(self) -> dict[str, Any]:
        attributes: dict[str, AttributeValue] = {
            "service.name": self.service_name,
            "service.instance.id": instance_id(),
            "telemetry.sdk.name": "linprogx-otlp",
            "telemetry.sdk.language": "python",
        }
        environment = _env("DEPLOY_ENVIRONMENT")
        if environment:
            attributes["deployment.environment.name"] = environment
        return {"attributes": _attributes(attributes)}

    def _send(self, batch: list[dict[str, Any]]) -> None:
        url = _traces_endpoint()
        if not url:
            return
        payload = {
            "resourceSpans": [
                {
                    "resource": self._resource(),
                    "scopeSpans": [
                        {
                            "scope": {"name": "linprogx.demo", "version": "1"},
                            "spans": batch,
                        }
                    ],
                }
            ]
        }
        headers = {"Content-Type": "application/json"}
        authorization = _env("OTEL_EXPORTER_OTLP_AUTHORIZATION")
        if authorization:
            headers["Authorization"] = authorization
        try:
            request = urllib.request.Request(  # noqa: S310 -- scheme validated above
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=_EXPORT_TIMEOUT_S) as response:  # noqa: S310
                if response.status >= 300:
                    self._note_failure()
        except Exception:  # noqa: BLE001 -- telemetry must never break a request
            self._note_failure()

    def _note_failure(self) -> None:
        """Count a dropped export, logging at most once a minute.

        The exception, URL, and credentials are deliberately never logged.
        """
        with self._failure_lock:
            self._failures += 1
            now = time.monotonic()
            if now - self._last_failure_log < _FAILURE_LOG_INTERVAL_S:
                return
            self._last_failure_log = now
            dropped, self._failures = self._failures, 0
        logger.warning("otel: dropped %d span export(s); tracing is fail-open", dropped)


# --- Spans -------------------------------------------------------------------

_current_span: ContextVar[Span | None] = ContextVar("linprogx_current_span", default=None)


class Span:
    """One recorded operation. Ending it hands an OTLP span to the exporter."""

    __slots__ = (
        "_attributes",
        "_ended",
        "_events",
        "_exporter",
        "_status",
        "_status_message",
        "context",
        "kind",
        "name",
        "parent_span_id",
        "sampled",
        "start_time_ns",
    )

    def __init__(
        self,
        name: str,
        kind: int,
        context: SpanContext,
        parent_span_id: str | None,
        sampled: bool,
        exporter: SpanExporter | None,
        start_time_ns: int,
    ) -> None:
        self.name = name
        self.kind = kind
        self.context = context
        self.parent_span_id = parent_span_id
        self.sampled = sampled
        self.start_time_ns = start_time_ns
        self._exporter = exporter
        self._attributes: MutableMapping[str, AttributeValue] = {}
        self._events: list[dict[str, Any]] = []
        self._status = STATUS_UNSET
        self._status_message = ""
        self._ended = False

    def set_attribute(self, key: str, value: AttributeValue | None) -> None:
        if value is None or not self.sampled:
            return
        if key not in self._attributes and len(self._attributes) >= _MAX_ATTRS:
            return
        self._attributes[key] = value

    def set_attributes(self, values: Mapping[str, AttributeValue | None]) -> None:
        for key, value in values.items():
            self.set_attribute(key, value)

    def set_status(self, code: int, message: str = "") -> None:
        self._status = code
        self._status_message = message[:_MAX_ATTR_CHARS]

    def add_event(self, name: str, attributes: Mapping[str, AttributeValue] | None = None) -> None:
        if not self.sampled:
            return
        self._events.append(
            {
                "timeUnixNano": str(time.time_ns()),
                "name": name[:_MAX_ATTR_CHARS],
                "attributes": _attributes(attributes or {}),
            }
        )

    def record_exception(self, exc: BaseException, *, escaped: bool = True) -> None:
        """Record the exception's *type* only.

        Messages and stack traces are omitted on purpose: a validation error can
        echo request content, and no exception text is worth the PII risk.
        """
        exception_type = type(exc).__name__
        self.add_event(
            "exception",
            {"exception.type": exception_type, "exception.escaped": escaped},
        )
        self.set_attribute("error.type", exception_type)

    def end(self, end_time_ns: int | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        if not self.sampled or self._exporter is None:
            return
        otlp: dict[str, Any] = {
            "traceId": self.context.trace_id,
            "spanId": self.context.span_id,
            "name": self.name[:_MAX_ATTR_CHARS],
            "kind": self.kind,
            "flags": int(self.context.trace_flags, 16),
            "startTimeUnixNano": str(self.start_time_ns),
            "endTimeUnixNano": str(end_time_ns if end_time_ns is not None else time.time_ns()),
            "attributes": _attributes(self._attributes),
            "status": (
                {"code": self._status, "message": self._status_message}
                if self._status_message
                else {"code": self._status}
            ),
        }
        if self.parent_span_id:
            otlp["parentSpanId"] = self.parent_span_id
        if self._events:
            otlp["events"] = self._events
        self._exporter.submit(otlp)


class Tracer:
    """Creates spans against one service name and exporter."""

    def __init__(self, service_name: str, exporter: SpanExporter | None) -> None:
        self.service_name = service_name
        self.exporter = exporter

    def start_span(
        self,
        name: str,
        *,
        kind: int = KIND_INTERNAL,
        parent: SpanContext | None = None,
        attributes: Mapping[str, AttributeValue | None] | None = None,
        start_time_ns: int | None = None,
    ) -> Span:
        """Start a span, continuing ``parent`` when given.

        Parent resolution: the explicit ``parent`` wins, then the span currently
        active on this context, then a brand-new root trace. A parent that is
        valid but *not sampled* still produces a span object with the right
        identity -- it simply records nothing, honouring the upstream decision.
        """
        if parent is None:
            active = _current_span.get()
            parent = active.context if active is not None else None

        if parent is None:
            context = SpanContext(trace_id=secrets.token_hex(16), span_id=secrets.token_hex(8))
            parent_span_id = None
            sampled = True
        else:
            context = SpanContext(
                trace_id=parent.trace_id,
                span_id=secrets.token_hex(8),
                trace_flags=parent.trace_flags,
            )
            parent_span_id = parent.span_id
            sampled = parent.sampled

        span = Span(
            name=name,
            kind=kind,
            context=context,
            parent_span_id=parent_span_id,
            sampled=sampled,
            exporter=self.exporter,
            start_time_ns=start_time_ns if start_time_ns is not None else time.time_ns(),
        )
        if attributes:
            span.set_attributes(attributes)
        return span

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: int = KIND_INTERNAL,
        parent: SpanContext | None = None,
        attributes: Mapping[str, AttributeValue | None] | None = None,
    ) -> Iterator[Span]:
        """Run a block as a span, making it the active parent for nested spans."""
        span = self.start_span(name, kind=kind, parent=parent, attributes=attributes)
        token = _current_span.set(span)
        try:
            yield span
        except _CANCELLED:
            # Cancellation is the caller going away, not this span failing.
            raise
        except BaseException as exc:
            span.record_exception(exc)
            span.set_status(STATUS_ERROR)
            raise
        finally:
            _current_span.reset(token)
            span.end()


def current_span() -> Span | None:
    """The span active on this context, if any."""
    return _current_span.get()


def current_span_context() -> SpanContext | None:
    """The active span's context, for handing a parent across a thread boundary.

    ``ContextVar`` values do not follow work submitted to a ``ThreadPoolExecutor``,
    so the solve path captures this on the request thread and passes it in.
    """
    span = _current_span.get()
    return span.context if span is not None else None


# --- Module-level configuration ----------------------------------------------

_TRACER: Tracer | None = None
_configure_lock = threading.Lock()


def configure(service_name: str) -> Tracer:
    """Install the process-wide tracer. Idempotent; safe to call at import."""
    global _TRACER
    with _configure_lock:
        if _TRACER is None:
            exporter = SpanExporter(service_name)
            _TRACER = Tracer(service_name, exporter)
            atexit.register(exporter.flush)
        return _TRACER


def get_tracer() -> Tracer:
    """The configured tracer, or an inert one if ``configure`` was never called."""
    return _TRACER if _TRACER is not None else Tracer("unconfigured", None)


def reset_for_tests() -> None:
    """Drop the process-wide tracer. Test-only."""
    global _TRACER
    with _configure_lock:
        _TRACER = None
    reset_cold_start_for_tests()


# --- ASGI middleware ---------------------------------------------------------


def _outcome(status_code: int) -> str:
    """Map a status onto the site's bounded outcome vocabulary."""
    if status_code < 400:
        return "success"
    if status_code == 404:
        return "not_found"
    if status_code == 422:
        return "invalid_request"
    if status_code == 429:
        return "rate_limited"
    if status_code == 504:
        return "timeout"
    if status_code < 500:
        return "client_error"
    return "internal_error"


def _header(scope: Mapping[str, Any], name: bytes) -> str | None:
    for key, value in scope.get("headers", ()):
        if key == name:
            try:
                return value.decode("latin-1").strip()
            except Exception:  # noqa: BLE001 -- a malformed header is simply absent
                return None
    return None


_KNOWN_METHODS = frozenset(
    {"CONNECT", "DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT", "TRACE"}
)


def _method(value: object) -> str:
    """Keep the HTTP method dimension finite even for scanner traffic."""
    candidate = str(value).upper()
    return candidate if candidate in _KNOWN_METHODS else "_OTHER"


class TracingMiddleware:
    """Pure-ASGI server-span middleware.

    Pure ASGI rather than ``BaseHTTPMiddleware`` so it composes cleanly with the
    app's own auth and rate-limit middleware and never buffers a response body.

    ``routes`` bounds ``http.route`` cardinality -- anything unrecognised is
    reported as ``/*`` so a scanner cannot mint unbounded span names.
    """

    def __init__(self, app: Any, *, routes: frozenset[str]) -> None:
        self.app = app
        self.routes = routes

    def _route(self, path: str) -> str:
        return path if path in self.routes else "/*"

    async def __call__(self, scope: MutableMapping[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        tracer = get_tracer()
        parent = parse_traceparent(_header(scope, b"traceparent"))
        method = _method(scope.get("method", "GET"))
        route = self._route(str(scope.get("path", "")))

        span = tracer.start_span(f"{method} {route}", kind=KIND_SERVER, parent=parent)
        span.set_attributes(
            {
                "http.request.method": method,
                "http.route": route,
                "url.scheme": str(scope.get("scheme", "http")),
                "network.protocol.version": str(scope.get("http_version", "1.1")),
                "faas.instance": instance_id(),
            }
        )

        correlation_id = _header(scope, b"x-correlation-id")
        if correlation_id and _CORRELATION_ID.match(correlation_id):
            span.set_attribute("request.correlation_id", correlation_id.lower())
        if _header(scope, b"x-request-purpose") == "warmup":
            span.set_attribute("request.purpose", "warmup")

        cold, reuse_bucket, init_seconds = request_container_state()
        span.set_attributes(
            {
                "faas.coldstart": cold,
                "container.reused": not cold,
                "container.reuse_bucket": reuse_bucket,
            }
        )
        if cold and init_seconds is not None:
            self._record_init_span(tracer, span, init_seconds)

        status_code = 500
        token = _current_span.set(span)

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except _CANCELLED as exc:
            # The client went away mid-request (navigation, refresh, an aborted
            # fetch). That is an expected outcome, not an incident: the span
            # status stays UNSET so it never fires a 5xx-shaped alert.
            span.set_attribute("request.outcome", "cancelled")
            raise exc from None
        except BaseException as exc:
            span.record_exception(exc)
            span.set_status(STATUS_ERROR)
            span.set_attributes(
                {"http.response.status_code": 500, "request.outcome": "internal_error"}
            )
            raise
        else:
            span.set_attributes(
                {
                    "http.response.status_code": status_code,
                    "request.outcome": _outcome(status_code),
                }
            )
            # OTel leaves successful server spans UNSET; only 5xx is an error.
            if status_code >= 500:
                span.set_status(STATUS_ERROR)
                span.set_attribute("error.type", str(status_code))
        finally:
            _current_span.reset(token)
            span.end()

    @staticmethod
    def _record_init_span(tracer: Tracer, request_span: Span, init_seconds: float) -> None:
        """Emit the container's init window as a child of the first request.

        The init span deliberately *starts before its parent*: it covers image
        import and interpreter startup, which finished just before this request
        arrived. Attaching it to the first request is what makes a cold start
        visible in the same waterfall as the user-visible latency it caused.
        """
        started_at = request_span.start_time_ns - int(init_seconds * 1_000_000_000)
        init_span = tracer.start_span(
            "container.init",
            parent=request_span.context,
            start_time_ns=started_at,
            attributes={"faas.coldstart": True, "faas.instance": instance_id()},
        )
        init_span.end(end_time_ns=request_span.start_time_ns)
