"""End-to-end coverage for the demo API's OTLP tracing.

These tests drive the real exporter -- queue, worker thread, JSON encoding --
with only the HTTP transport replaced, so a payload a collector would reject
fails here too. The middleware is exercised through the raw ASGI protocol rather
than a web test client, which keeps the solver repo free of a web test stack and
tests the actual contract the middleware implements.

The load-bearing properties, in order of how badly a regression would hurt:

1. A valid inbound ``traceparent`` continues the Cloudflare proxy's trace.
2. A missing or malformed one starts a fresh trace instead of erroring.
3. Nothing telemetry does can fail a user request -- including a collector that
   is down, hanging, or returning 500.
4. Nothing recorded identifies a caller or the problem they submitted.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from demo.api import tracing

VALID_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
VALID_SPAN_ID = "00f067aa0ba902b7"
VALID_TRACEPARENT = f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-01"

ROUTES = frozenset({"/api/health", "/api/info", "/api/solve/network-flow"})


# --- Collector ---------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> bool:
        return False


class _Collector:
    """Stands in for the OTLP receiver; records every exported span."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.requests: list[Any] = []
        self.timeouts: list[float | None] = []
        self.status = 200
        self.error: Exception | None = None
        self.delay = 0.0

    def urlopen(self, request: Any, timeout: float | None = None) -> Any:
        self.requests.append(request)
        self.timeouts.append(timeout)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        self.payloads.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse(self.status)

    @property
    def spans(self) -> list[dict[str, Any]]:
        return [
            span
            for payload in self.payloads
            for resource_span in payload["resourceSpans"]
            for scope_span in resource_span["scopeSpans"]
            for span in scope_span["spans"]
        ]

    def resource_attrs(self) -> dict[str, Any]:
        return _flatten(self.payloads[0]["resourceSpans"][0]["resource"]["attributes"])

    def named(self, name: str) -> dict[str, Any] | None:
        return next((span for span in self.spans if span["name"] == name), None)

    def await_named(self, name: str, timeout: float = 5.0) -> dict[str, Any]:
        """Block until a span with this name is exported.

        Spans arrive in batches, so "the first span I see" is not a safe way to
        find one.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            span = self.named(name)
            if span is not None:
                return span
            time.sleep(0.02)
        raise AssertionError(f"no {name!r} span exported; saw {[s['name'] for s in self.spans]}")

    def settle(self, seconds: float = 0.4) -> None:
        """Wait long enough that a batch would have been scheduled and sent."""
        time.sleep(seconds)


def _flatten(attributes: list[dict[str, Any]]) -> dict[str, Any]:
    """OTLP attribute list -> plain dict, unwrapping the typed value envelope."""
    out: dict[str, Any] = {}
    for attribute in attributes:
        ((kind, value),) = attribute["value"].items()
        out[attribute["key"]] = int(value) if kind == "intValue" else value
    return out


def _attrs(span: dict[str, Any]) -> dict[str, Any]:
    return _flatten(span.get("attributes", []))


# --- Minimal ASGI harness ----------------------------------------------------


def _make_app(
    *,
    status: int = 200,
    exc: BaseException | None = None,
    exc_after_start: BaseException | None = None,
    inner_span: str | None = None,
) -> Any:
    """A stand-in for the FastAPI app: the middleware only sees the protocol."""

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if inner_span is not None:
            # Mirrors how the real endpoint opens its solve span: no explicit
            # parent, relying on the middleware's active context.
            with tracing.get_tracer().span(inner_span) as span:
                span.set_attribute("lp.nodes", 3)
        if exc is not None:
            raise exc
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        if exc_after_start is not None:
            raise exc_after_start
        await send({"type": "http.response.body", "body": b'{"status":"ok"}'})

    return app


def _request(
    app: Any,
    *,
    method: str = "GET",
    path: str = "/api/health",
    headers: dict[str, str] | None = None,
    scope_type: str = "http",
) -> list[dict[str, Any]]:
    """Drive one request through an ASGI app, returning the sent messages."""
    scope: dict[str, Any] = {
        "type": scope_type,
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "https",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": ("203.0.113.7", 51234),
        "server": ("linprogx.modal.run", 443),
    }
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent


@pytest.fixture
def collector(monkeypatch: pytest.MonkeyPatch) -> _Collector:
    """Point the process-wide exporter at an in-memory collector."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.example/otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_AUTHORIZATION", "Basic dGVzdA==")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    sink = _Collector()
    monkeypatch.setattr(tracing.urllib.request, "urlopen", sink.urlopen)
    tracing.configure("linprogx")
    # Every test gets a fresh cold start so the marker is deterministic.
    tracing.reset_cold_start_for_tests()
    return sink


@pytest.fixture
def traced() -> Any:
    """The middleware wrapped around a healthy stand-in app."""
    return tracing.TracingMiddleware(_make_app(), routes=ROUTES)


# --- W3C traceparent parsing -------------------------------------------------


def test_parse_traceparent_accepts_a_valid_header() -> None:
    context = tracing.parse_traceparent(VALID_TRACEPARENT)

    assert context is not None
    assert context.trace_id == VALID_TRACE_ID
    assert context.span_id == VALID_SPAN_ID
    assert context.sampled is True


def test_parse_traceparent_reads_the_unsampled_flag() -> None:
    context = tracing.parse_traceparent(f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-00")

    assert context is not None
    assert context.sampled is False


def test_span_context_round_trips_to_a_traceparent() -> None:
    context = tracing.parse_traceparent(VALID_TRACEPARENT)

    assert context is not None
    assert context.traceparent() == VALID_TRACEPARENT


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("missing", None),
        ("empty", ""),
        ("whitespace", "   "),
        ("no separators", "not-a-traceparent"),
        ("too few fields", f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}"),
        ("all-zero trace id", f"00-{'0' * 32}-{VALID_SPAN_ID}-01"),
        ("all-zero span id", f"00-{VALID_TRACE_ID}-{'0' * 16}-01"),
        ("short trace id", f"00-{'a' * 31}-{VALID_SPAN_ID}-01"),
        ("long trace id", f"00-{'a' * 33}-{VALID_SPAN_ID}-01"),
        ("non-hex trace id", f"00-{'z' * 32}-{VALID_SPAN_ID}-01"),
        ("short span id", f"00-{VALID_TRACE_ID}-{'a' * 15}-01"),
        ("non-hex flags", f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-zz"),
        ("uppercase", VALID_TRACEPARENT.upper()),
        ("leading whitespace", f" {VALID_TRACEPARENT}"),
        ("forbidden version ff", f"ff-{VALID_TRACE_ID}-{VALID_SPAN_ID}-01"),
        ("v00 with extra field", f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-01-extra"),
        ("future version, missing extension dash", f"01-{VALID_TRACE_ID}-{VALID_SPAN_ID}-01x"),
        ("future version, control character", f"01-{VALID_TRACE_ID}-{VALID_SPAN_ID}-01-\n"),
        ("oversized", "00-" + "a" * 600),
    ],
)
def test_parse_traceparent_rejects_invalid_headers(label: str, value: str | None) -> None:
    assert tracing.parse_traceparent(value) is None, label


@pytest.mark.parametrize(("flags", "expected"), [("03", "01"), ("02", "00"), ("ff", "01")])
def test_parse_traceparent_masks_reserved_flag_bits(flags: str, expected: str) -> None:
    context = tracing.parse_traceparent(f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-{flags}")

    assert context is not None
    assert context.trace_flags == expected


@pytest.mark.parametrize(
    "value",
    [
        f"01-{VALID_TRACE_ID}-{VALID_SPAN_ID}-01",
        f"01-{VALID_TRACE_ID}-{VALID_SPAN_ID}-01-vendor-fields",
        f"fe-{VALID_TRACE_ID}-{VALID_SPAN_ID}-03-opaque-extension",
        f"01-{VALID_TRACE_ID}-{VALID_SPAN_ID}-00-",
    ],
)
def test_parse_traceparent_accepts_the_known_prefix_of_future_versions(value: str) -> None:
    context = tracing.parse_traceparent(value)

    assert context is not None
    assert context.trace_id == VALID_TRACE_ID
    assert context.span_id == VALID_SPAN_ID
    assert context.trace_flags in {"00", "01"}


# --- Trace continuation ------------------------------------------------------


def test_valid_context_makes_the_server_span_a_child_of_the_edge_span(
    collector: _Collector, traced: Any
) -> None:
    _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    server = collector.await_named("GET /api/health")
    assert server["traceId"] == VALID_TRACE_ID
    assert server["parentSpanId"] == VALID_SPAN_ID
    assert server["kind"] == tracing.KIND_SERVER


def test_future_context_keeps_the_server_span_in_the_upstream_trace(
    collector: _Collector, traced: Any
) -> None:
    future = f"01-{VALID_TRACE_ID}-{VALID_SPAN_ID}-03-opaque"

    _request(traced, headers={"traceparent": future})

    server = collector.await_named("GET /api/health")
    assert server["traceId"] == VALID_TRACE_ID
    assert server["parentSpanId"] == VALID_SPAN_ID
    assert server["flags"] == 1


def test_missing_context_starts_a_new_root_trace(collector: _Collector, traced: Any) -> None:
    _request(traced)

    server = collector.await_named("GET /api/health")
    assert "parentSpanId" not in server
    assert len(server["traceId"]) == 32
    assert server["traceId"] != VALID_TRACE_ID


def test_invalid_context_starts_a_new_root_trace(collector: _Collector, traced: Any) -> None:
    _request(traced, headers={"traceparent": "00-garbage-nonsense-01"})

    server = collector.await_named("GET /api/health")
    assert "parentSpanId" not in server
    assert len(server["traceId"]) == 32


def test_two_requests_without_context_get_different_traces(
    collector: _Collector, traced: Any
) -> None:
    _request(traced)
    first = collector.await_named("GET /api/health")["traceId"]
    collector.payloads.clear()
    _request(traced)

    assert collector.await_named("GET /api/health")["traceId"] != first


def test_unsampled_parent_suppresses_export(collector: _Collector, traced: Any) -> None:
    """An upstream sampling decision of "no" is honoured, not overridden."""
    _request(traced, headers={"traceparent": f"00-{VALID_TRACE_ID}-{VALID_SPAN_ID}-00"})

    collector.settle()
    assert collector.spans == []


def test_non_http_scopes_pass_through_untouched(collector: _Collector, traced: Any) -> None:
    async def lifespan(scope: dict[str, Any], receive: Any, send: Any) -> None:
        await send({"type": "lifespan.startup.complete"})

    middleware = tracing.TracingMiddleware(lifespan, routes=ROUTES)
    sent = _request(middleware, scope_type="lifespan")

    assert sent == [{"type": "lifespan.startup.complete"}]
    collector.settle()
    assert collector.spans == []


# --- Fail-open ---------------------------------------------------------------


def test_collector_connection_failure_never_reaches_the_user(
    collector: _Collector, traced: Any
) -> None:
    collector.error = OSError("connection refused")

    sent = _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    assert sent[0]["status"] == 200
    assert sent[1]["body"] == b'{"status":"ok"}'
    collector.settle()
    assert collector.payloads == []


def test_collector_rejection_never_reaches_the_user(collector: _Collector, traced: Any) -> None:
    collector.status = 500

    sent = _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    assert sent[0]["status"] == 200
    collector.settle()


def test_a_hanging_collector_never_blocks_the_response(collector: _Collector, traced: Any) -> None:
    """Export runs off-thread, so a slow collector cannot slow a request down."""
    collector.delay = 1.0

    started = time.monotonic()
    sent = _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    assert sent[0]["status"] == 200
    assert time.monotonic() - started < 0.5


def test_unconfigured_endpoint_exports_nothing_and_still_serves(
    monkeypatch: pytest.MonkeyPatch, traced: Any
) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    sink = _Collector()
    monkeypatch.setattr(tracing.urllib.request, "urlopen", sink.urlopen)
    tracing.configure("linprogx")

    sent = _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    assert sent[0]["status"] == 200
    sink.settle()
    assert sink.payloads == []


def test_sdk_disabled_switch_exports_nothing(
    monkeypatch: pytest.MonkeyPatch, collector: _Collector, traced: Any
) -> None:
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")

    _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    collector.settle()
    assert collector.payloads == []


def test_non_http_endpoint_is_refused(
    monkeypatch: pytest.MonkeyPatch, collector: _Collector, traced: Any
) -> None:
    """A file:// endpoint must never turn telemetry into a local-file write."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "file:///tmp/otlp")

    _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    collector.settle()
    assert collector.payloads == []


def test_authenticated_plain_http_endpoint_is_refused(
    monkeypatch: pytest.MonkeyPatch, collector: _Collector, traced: Any
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.example/otlp")

    _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    collector.settle()
    assert collector.payloads == []


def test_unauthenticated_plain_http_endpoint_is_allowed_for_local_collectors(
    monkeypatch: pytest.MonkeyPatch, collector: _Collector, traced: Any
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_AUTHORIZATION")

    _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    collector.await_named("GET /api/health")


def test_an_explicit_traces_endpoint_overrides_the_base(
    monkeypatch: pytest.MonkeyPatch, collector: _Collector, traced: Any
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "https://collector.example/custom")

    _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    collector.await_named("GET /api/health")


def test_export_uses_the_otlp_json_endpoint_headers_and_encoding(
    collector: _Collector, traced: Any
) -> None:
    _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    span = collector.await_named("GET /api/health")
    request = collector.requests[0]
    assert request.full_url == "https://collector.example/otlp/v1/traces"
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("Authorization") == "Basic dGVzdA=="
    assert collector.timeouts[0] == tracing._EXPORT_TIMEOUT_S
    # OTLP/HTTP JSON represents trace/span byte fields as lowercase hex and
    # uint32 trace flags as a JSON number.
    assert span["traceId"] == VALID_TRACE_ID
    assert span["spanId"] == span["spanId"].lower()
    assert len(span["spanId"]) == 16
    assert span["flags"] == 1
    int(span["startTimeUnixNano"])
    int(span["endTimeUnixNano"])


def test_flush_waits_for_an_in_flight_export(collector: _Collector) -> None:
    collector.delay = 0.2
    tracer = tracing.configure("linprogx")
    span = tracer.start_span("flush.probe")
    span.end()

    assert tracer.exporter is not None
    tracer.exporter.flush(timeout=1.0)

    assert collector.named("flush.probe") is not None


def test_lifespan_shutdown_flushes_an_in_flight_export(collector: _Collector) -> None:
    collector.delay = 0.2
    tracer = tracing.configure("linprogx")
    span = tracer.start_span("shutdown.probe")
    span.end()

    async def lifespan(scope: dict[str, Any], receive: Any, send: Any) -> None:
        await send({"type": "lifespan.shutdown.complete"})

    middleware = tracing.TracingMiddleware(lifespan, routes=ROUTES)
    sent = _request(middleware, scope_type="lifespan")

    assert sent == [{"type": "lifespan.shutdown.complete"}]
    assert collector.named("shutdown.probe") is not None


def test_lifespan_shutdown_honors_the_flush_deadline_without_an_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = tracing.SpanExporter("linprogx")
    # Model an export that never completes without starting another thread.
    exporter._pending = 1
    monkeypatch.setattr(tracing, "_TRACER", tracing.Tracer("linprogx", exporter))
    monkeypatch.setattr(tracing, "_SHUTDOWN_FLUSH_TIMEOUT_S", 0.05)

    async def lifespan(scope: dict[str, Any], receive: Any, send: Any) -> None:
        await send({"type": "lifespan.shutdown.complete"})

    middleware = tracing.TracingMiddleware(lifespan, routes=ROUTES)
    started = time.monotonic()
    sent = _request(middleware, scope_type="lifespan")
    elapsed = time.monotonic() - started

    assert sent == [{"type": "lifespan.shutdown.complete"}]
    assert 0.04 <= elapsed < 0.25


def test_id_generation_failure_never_reaches_the_user(
    monkeypatch: pytest.MonkeyPatch, collector: _Collector, traced: Any
) -> None:
    def unavailable(*_: object) -> str:
        raise OSError

    monkeypatch.setattr(tracing.secrets, "token_hex", unavailable)

    sent = _request(traced)

    assert sent[0]["status"] == 200


def test_exporter_thread_start_failure_never_reaches_the_user(
    monkeypatch: pytest.MonkeyPatch, collector: _Collector, traced: Any
) -> None:
    exporter = tracing.SpanExporter("linprogx")
    monkeypatch.setattr(tracing, "_TRACER", tracing.Tracer("linprogx", exporter))

    def unavailable(*_: object) -> None:
        raise RuntimeError("threads unavailable")

    monkeypatch.setattr(tracing.threading.Thread, "start", unavailable)

    sent = _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    assert sent[0]["status"] == 200


def test_span_submission_failure_never_reaches_the_user(
    monkeypatch: pytest.MonkeyPatch, collector: _Collector, traced: Any
) -> None:
    exporter = tracing.SpanExporter("linprogx")
    monkeypatch.setattr(tracing, "_TRACER", tracing.Tracer("linprogx", exporter))

    def unavailable(*_: object) -> None:
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(exporter, "submit", unavailable)

    sent = _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    assert sent[0]["status"] == 200


def test_queue_drop_logging_failure_never_reaches_the_user(
    monkeypatch: pytest.MonkeyPatch, collector: _Collector, traced: Any
) -> None:
    exporter = tracing.SpanExporter("linprogx")
    exporter._queue = tracing.queue.Queue(maxsize=1)
    exporter._queue.put_nowait({"already": "full"})
    exporter._last_failure_log = -float("inf")
    monkeypatch.setattr(exporter, "_ensure_worker", lambda: None)
    monkeypatch.setattr(tracing, "_TRACER", tracing.Tracer("linprogx", exporter))

    def unavailable(*_: object) -> None:
        raise RuntimeError("logging unavailable")

    monkeypatch.setattr(tracing.logger, "warning", unavailable)

    sent = _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    assert sent[0]["status"] == 200


# --- Request span content ----------------------------------------------------


def test_server_span_carries_bounded_http_attributes(collector: _Collector, traced: Any) -> None:
    _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    attributes = _attrs(collector.await_named("GET /api/health"))
    assert attributes["http.request.method"] == "GET"
    assert attributes["http.route"] == "/api/health"
    assert attributes["http.response.status_code"] == 200
    assert attributes["request.outcome"] == "success"
    assert attributes["url.scheme"] == "https"


def test_unknown_methods_collapse_to_one_bounded_value(collector: _Collector, traced: Any) -> None:
    """Scanner traffic must not mint an unbounded http.request.method dimension."""
    _request(traced, method="BREW", headers={"traceparent": VALID_TRACEPARENT})

    span = collector.await_named("_OTHER /api/health")
    assert _attrs(span)["http.request.method"] == "_OTHER"


def test_unknown_routes_collapse_to_a_single_bounded_name(
    collector: _Collector, traced: Any
) -> None:
    _request(traced, path="/api/../etc/passwd", headers={"traceparent": VALID_TRACEPARENT})

    span = collector.await_named("GET /*")
    assert _attrs(span)["http.route"] == "/*"


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        (200, "success"),
        (401, "client_error"),
        (404, "not_found"),
        (429, "rate_limited"),
        (500, "internal_error"),
    ],
)
def test_status_codes_map_onto_bounded_outcomes(
    collector: _Collector, status: int, outcome: str
) -> None:
    middleware = tracing.TracingMiddleware(_make_app(status=status), routes=ROUTES)

    _request(middleware, headers={"traceparent": VALID_TRACEPARENT})

    attributes = _attrs(collector.await_named("GET /api/health"))
    assert attributes["request.outcome"] == outcome
    assert attributes["http.response.status_code"] == status


def test_only_server_errors_set_an_error_span_status(collector: _Collector) -> None:
    """A 401 or 429 is the API working correctly; it must not look like an incident."""
    for status, expected in ((429, tracing.STATUS_UNSET), (500, tracing.STATUS_ERROR)):
        collector.payloads.clear()
        tracing.reset_cold_start_for_tests()
        middleware = tracing.TracingMiddleware(_make_app(status=status), routes=ROUTES)

        _request(middleware, headers={"traceparent": VALID_TRACEPARENT})

        assert collector.await_named("GET /api/health")["status"]["code"] == expected


def test_warmup_requests_are_labelled(collector: _Collector, traced: Any) -> None:
    _request(
        traced,
        headers={"traceparent": VALID_TRACEPARENT, "x-request-purpose": "warmup"},
    )

    assert _attrs(collector.await_named("GET /api/health"))["request.purpose"] == "warmup"


def test_caller_provided_correlation_ids_are_never_recorded(
    collector: _Collector, traced: Any
) -> None:
    _request(
        traced,
        headers={
            "traceparent": VALID_TRACEPARENT,
            "x-correlation-id": "0af7651916cd43dd8448eb211c80319c",
        },
    )

    assert "request.correlation_id" not in _attrs(collector.await_named("GET /api/health"))


def test_an_unhandled_exception_is_recorded_by_type_and_re_raised(
    collector: _Collector,
) -> None:
    middleware = tracing.TracingMiddleware(
        _make_app(exc=RuntimeError("secret detail from the request")), routes=ROUTES
    )

    with pytest.raises(RuntimeError):
        _request(middleware, headers={"traceparent": VALID_TRACEPARENT})

    span = collector.await_named("GET /api/health")
    assert span["status"]["code"] == tracing.STATUS_ERROR
    attributes = _attrs(span)
    assert attributes["error.type"] == "RuntimeError"
    assert attributes["request.outcome"] == "internal_error"
    # The exception is recorded by type only -- never its message.
    assert "secret detail" not in json.dumps(span)


def test_late_stream_error_preserves_the_status_already_sent(
    collector: _Collector,
) -> None:
    middleware = tracing.TracingMiddleware(
        _make_app(exc_after_start=RuntimeError("late stream failure")),
        routes=ROUTES,
    )

    with pytest.raises(RuntimeError):
        _request(middleware, headers={"traceparent": VALID_TRACEPARENT})

    span = collector.await_named("GET /api/health")
    attributes = _attrs(span)
    assert attributes["http.response.status_code"] == 200
    assert attributes["request.outcome"] == "internal_error"
    assert span["status"]["code"] == tracing.STATUS_ERROR


def test_resource_identifies_the_service_and_container(collector: _Collector, traced: Any) -> None:
    _request(traced, headers={"traceparent": VALID_TRACEPARENT})
    collector.await_named("GET /api/health")

    resource = collector.resource_attrs()
    assert resource["service.name"] == "linprogx"
    assert resource["service.instance.id"] == tracing.instance_id()


def test_attribute_values_are_truncated(collector: _Collector) -> None:
    tracer = tracing.configure("linprogx")
    with tracer.span("bounded") as span:
        span.set_attribute("long", "x" * 500)

    exported = collector.await_named("bounded")
    assert len(_attrs(exported)["long"]) == 128


# --- Nested phase spans ------------------------------------------------------


def test_a_nested_span_becomes_a_child_of_the_request_span(collector: _Collector) -> None:
    """The solve span finds its parent through the active context, not an argument."""
    middleware = tracing.TracingMiddleware(
        _make_app(inner_span="solve.network_flow"), routes=ROUTES
    )

    _request(
        middleware,
        path="/api/solve/network-flow",
        method="POST",
        headers={"traceparent": VALID_TRACEPARENT},
    )

    server = collector.await_named("POST /api/solve/network-flow")
    solve = collector.await_named("solve.network_flow")
    assert solve["traceId"] == VALID_TRACE_ID
    assert solve["parentSpanId"] == server["spanId"]
    assert _attrs(solve)["lp.nodes"] == 3


def test_an_explicit_parent_crosses_a_thread_boundary(collector: _Collector) -> None:
    """ContextVars do not follow work onto a pool thread; a passed context does."""
    tracer = tracing.configure("linprogx")
    parent = tracer.start_span("carrier")
    child = tracer.start_span("worker", parent=parent.context)
    child.end()
    parent.end()

    exported = collector.await_named("worker")
    assert exported["parentSpanId"] == parent.context.span_id
    assert exported["traceId"] == parent.context.trace_id


# --- Cold start --------------------------------------------------------------


def test_first_request_reports_a_cold_container(collector: _Collector, traced: Any) -> None:
    _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    server = collector.await_named("GET /api/health")

    assert _attrs(server)["faas.coldstart"] is True
    assert _attrs(server)["container.reused"] is False
    assert _attrs(server)["container.reuse_bucket"] == "first"
    assert all(span["name"] != "container.init" for span in collector.spans)


def test_later_requests_report_a_warm_container(collector: _Collector, traced: Any) -> None:
    _request(traced, headers={"traceparent": VALID_TRACEPARENT})
    collector.await_named("GET /api/health")
    collector.payloads.clear()

    _request(traced, headers={"traceparent": VALID_TRACEPARENT})

    server = collector.await_named("GET /api/health")
    assert _attrs(server)["faas.coldstart"] is False
    assert _attrs(server)["container.reused"] is True
    assert _attrs(server)["container.reuse_bucket"] == "2-5"
    assert all(span["name"] != "container.init" for span in collector.spans)


def test_cold_start_marker_is_claimed_exactly_once() -> None:
    tracing.reset_cold_start_for_tests()

    first, first_bucket = tracing.request_container_state()
    second, second_bucket = tracing.request_container_state()

    assert (first, second) == (True, False)
    assert (first_bucket, second_bucket) == ("first", "2-5")


def test_instance_id_changes_after_a_snapshot_restore_identity_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Modal snapshots module globals, so the id must not be minted at import."""
    monkeypatch.setattr(tracing.socket, "gethostname", lambda: "restored-container-a")
    first = tracing.instance_id()
    assert tracing.instance_id() == first

    monkeypatch.setattr(tracing.socket, "gethostname", lambda: "restored-container-b")
    assert tracing.instance_id() != first


def test_instance_id_is_fail_open_when_runtime_identity_sources_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_: object) -> str:
        raise OSError

    monkeypatch.setattr(tracing.os, "getpid", lambda: 424_242)
    monkeypatch.setattr(tracing.socket, "gethostname", unavailable)
    monkeypatch.setattr(tracing.secrets, "token_hex", unavailable)

    identifier = tracing.instance_id()

    assert len(identifier) == 16
    int(identifier, 16)


# --- The real app ------------------------------------------------------------


def _real_request(
    app: Any,
    method: str,
    path: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any] | None = None,
) -> Any:
    """Drive the real FastAPI stack without Starlette's deprecated sync client."""
    import httpx

    async def run() -> Any:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, headers=headers, json=json_body)

    return asyncio.run(run())


def test_real_app_traces_a_network_flow_solve(collector: _Collector) -> None:
    from demo.api.main import app

    response = _real_request(
        app,
        "POST",
        "/api/solve/network-flow",
        json_body={
            "nodes": [
                {"id": "Warehouse Denver", "type": "supply", "value": 10},
                {"id": "Store Tulsa", "type": "demand", "value": 10},
            ],
            "edges": [{"from": "Warehouse Denver", "to": "Store Tulsa", "cost": 2, "capacity": 20}],
        },
        headers={"traceparent": VALID_TRACEPARENT},
    )

    assert response.status_code == 200
    server = collector.await_named("POST /api/solve/network-flow")
    solve = collector.await_named("solve.network_flow")
    assert server["traceId"] == VALID_TRACE_ID
    assert solve["parentSpanId"] == server["spanId"]

    attributes = _attrs(solve)
    assert attributes["solve.result"] == "optimal"
    assert attributes["lp.nodes"] == 2
    assert attributes["lp.edges"] == 1

    # No node name reaches the collector.
    exported = json.dumps(collector.payloads)
    assert "Denver" not in exported
    assert "Tulsa" not in exported


def test_real_app_traces_validation_failures(collector: _Collector) -> None:
    from demo.api.main import app

    response = _real_request(
        app,
        "POST",
        "/api/solve/network-flow",
        json_body={},
        headers={"traceparent": VALID_TRACEPARENT},
    )

    assert response.status_code == 422
    span = collector.await_named("POST /api/solve/network-flow")
    assert _attrs(span)["http.response.status_code"] == 422
    assert _attrs(span)["request.outcome"] == "invalid_request"


def test_real_app_traces_auth_short_circuit(
    monkeypatch: pytest.MonkeyPatch, collector: _Collector
) -> None:
    from demo.api import main

    monkeypatch.setattr(main, "_DEMO_SECRET", "expected")
    response = _real_request(
        main.app,
        "GET",
        "/api/info",
        headers={"traceparent": VALID_TRACEPARENT, "x-demo-secret": "wrong"},
    )

    assert response.status_code == 401
    span = collector.await_named("GET /api/info")
    assert _attrs(span)["http.response.status_code"] == 401
    assert _attrs(span)["request.outcome"] == "client_error"


def test_real_app_traces_rate_limit_short_circuit(
    monkeypatch: pytest.MonkeyPatch, collector: _Collector
) -> None:
    from demo.api import main

    monkeypatch.setattr(main, "_check_rate_limit", lambda client_ip: False)
    response = _real_request(
        main.app,
        "GET",
        "/api/health",
        headers={"traceparent": VALID_TRACEPARENT},
    )

    assert response.status_code == 429
    span = collector.await_named("GET /api/health")
    assert _attrs(span)["http.response.status_code"] == 429
    assert _attrs(span)["request.outcome"] == "rate_limited"


def test_a_client_disconnect_is_an_outcome_not_an_error(collector: _Collector) -> None:
    """A reader navigating away must not look like a server fault."""
    import asyncio

    middleware = tracing.TracingMiddleware(_make_app(exc=asyncio.CancelledError()), routes=ROUTES)

    with pytest.raises(asyncio.CancelledError):
        _request(middleware, headers={"traceparent": VALID_TRACEPARENT})

    span = collector.await_named("GET /api/health")
    assert span["status"]["code"] == tracing.STATUS_UNSET
    assert _attrs(span)["request.outcome"] == "cancelled"
    assert "error.type" not in _attrs(span)
