# Observability — linprogx demo API on Modal

The public demo already had spans from the browser (Grafana Faro) and from the
Cloudflare Pages Function that proxies `/api/lp/*`. This service is the third
leg: it continues the trace the proxy forwards, so a slow demo request is one
waterfall — browser fetch → edge → Modal → LP solve — instead of three unlinked
traces.

The site repository (`conway-personal-website`) owns the shared contract in
`docs/observability.md`; this document covers only what this repo implements.

## What emits spans

`demo/api/tracing.py` is a stdlib-only OTLP/HTTP client. It deliberately does
not use the OpenTelemetry SDK:

- this repo's contract keeps the runtime dependency-light, and the demo image is
  intentionally tiny so Modal cold starts stay fast; the SDK plus its FastAPI
  instrumentation adds import work to every cold start;
- the edge exporter (`functions/_observability.ts` in the site repo) is already
  a hand-rolled OTLP/HTTP JSON client, so matching it keeps one contract;
- it adds no runtime dependency. The existing `dev` extra includes the pinned
  FastAPI demo stack so CI always exercises the real middleware composition.

The wire format is standard OTLP/HTTP with a JSON payload, and attributes follow
OTel semantic conventions, so Grafana Cloud ingests it with no special handling.

The module lives under `demo/`, not `src/linprogx/`: it is demo-service
infrastructure, and the solver itself stays dependency-free and untouched.

## Configuration

| Variable | Required | Meaning |
| --- | --- | --- |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | to export anything | OTLP base URL, e.g. `https://otlp-gateway-prod-us-east-0.grafana.net/otlp`. `/v1/traces` is appended. |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | no | Full traces URL; overrides the base above. |
| `OTEL_EXPORTER_OTLP_AUTHORIZATION` | for Grafana Cloud | Complete `Authorization` header value, e.g. `Basic <base64 instance-id:token>`. Omit for an unauthenticated local collector. |
| `OTEL_SDK_DISABLED` | no | `true` turns export off without a redeploy path change. |
| `DEPLOY_ENVIRONMENT` | no | Sets `deployment.environment.name` on the resource. |

Authenticated export requires `https://`. Unauthenticated `http://` remains
available for a local collector; anything else is treated as unconfigured.

### Modal secret

`deploy/modal_app.py` attaches a Modal secret named **`otel-grafana`**, holding
`OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_AUTHORIZATION`. The same
secret backs the TourneyDesk demo app, so one token rotation covers both
services.

Create it once, before deploying either app — Modal resolves secrets by name at
deploy time and fails the deploy if the name does not exist:

```bash
modal secret create otel-grafana \
  OTEL_EXPORTER_OTLP_ENDPOINT='https://otlp-gateway-prod-us-east-0.grafana.net/otlp' \
  OTEL_EXPORTER_OTLP_AUTHORIZATION='Basic <base64 instance-id:token>'
```

Use a write-only Grafana Cloud Access Policy token. To turn tracing off without
touching code, replace the endpoint with an empty string (or set
`OTEL_SDK_DISABLED=true`) and redeploy; the service keeps serving either way.

## Span model

`service.name` is `linprogx`. `service.instance.id` is a random
per-container-process id, minted lazily after a Modal snapshot restore, so a
trace can be attributed to the actual serving container without recording
anything about the caller or cloning an import-time id across restored workers.
Inbound W3C version `00` context is validated as lowercase hexadecimal. For a
future version, the known trace id, parent id, and sampled bit are retained while
opaque trailing fields are ignored; this service emits its child context as
version `00`.

| Span | Kind | When |
| --- | --- | --- |
| `{METHOD} {route}` | server | Every HTTP request. Route is one of `/api/health`, `/api/info`, `/api/solve/network-flow`, or `/*`. |
| `solve.network_flow` | internal | The min-cost-flow LP run. |

Notable attributes:

- request: `http.request.method`, `http.route`, `http.response.status_code`,
  `request.outcome` (`success` / `not_found` / `invalid_request` /
  `rate_limited` / `timeout` / `client_error` / `internal_error` /
  `cancelled`),
  `request.purpose=warmup`, `faas.coldstart`, `faas.instance`,
  `container.reused`, and bounded `container.reuse_bucket`
  (`first` / `2-5` / `6-20` / `21+`).
- solve: `solve.result` (`optimal` / `infeasible` / `unbounded` /
  `iteration_limit` / `timeout` / `error`), `lp.nodes`, `lp.edges`,
  `lp.iterations`, `lp.solve_time_ms`.

`solve.result` is bounded by construction: `linprogx.Status` is a four-value
`StrEnum`, plus the two outcomes the endpoint adds itself.

5xx responses and unhandled server exceptions set span status `ERROR`. A late
stream exception preserves the status already sent to the client while still
marking the span as an error. Infeasible or unbounded results, a hit solve cap,
the 30 req/min rate limit, 4xx validation failures, and a client that disconnects
mid-request (`request.outcome=cancelled`) are recorded as outcomes, not incidents
— they are the API working correctly.

### Cold starts

The first request in each restored process reports `faas.coldstart=true`;
subsequent requests report `false`, `container.reused=true`, and a bounded reuse
bucket. No synthetic initialization duration is emitted: Modal snapshots module
globals, so an import-time clock can measure snapshot age rather than
serving-container startup. The request span and the upstream edge span provide
the real user-visible latency.

Container identity is minted lazily from runtime process identity rather than at
import, so one snapshotted id is not cloned into every restored container.

## Privacy

Never recorded: node or edge names, costs, capacities, solver request/response
bodies, headers, IP addresses, user agents, query strings, or exception
messages. Exceptions are recorded by **type** only, because a solver error can
echo request content.

Attribute values are truncated to 128 characters and capped at 32 per span.
`http.route` and `http.request.method` are drawn from fixed allowlists, and
container reuse is bucketed, so a scanner or long-lived container cannot mint
unbounded indexed values.

## Fail-open

No telemetry path can fail a user request:

- export runs on a bounded queue drained by a daemon thread, so a request never
  waits on the collector;
- a full queue drops spans;
- identifier generation, span encoding/end, thread creation, and logging-handler
  failures all degrade to inert telemetry;
- connection failures, timeouts, and non-2xx responses are swallowed and counted,
  logged at most once a minute, and never include the URL, credentials, or the
  exception;
- with no endpoint configured, the whole path costs a few dict lookups.

The exporter thread starts on first use rather than at import, keeping it out of
Modal's memory snapshot; a pid/liveness check restarts it after a snapshot
restore or fork. ASGI lifespan shutdown flushes pending exports before
acknowledging shutdown, with a bounded six-second budget that exceeds the
five-second HTTP export timeout. `atexit` uses the same budget as a fallback.

`tests/test_demo_tracing.py` covers all of this — valid, invalid, and missing
trace context, unsampled upstream decisions, collector connection failure,
collector 500, a hanging collector, unconfigured and disabled exporters, and the
privacy assertions. It drives the middleware through the raw ASGI protocol, so
fault injection, future-version propagation, streaming behavior, and bounded
lifespan shutdown are isolated from the framework. The pinned development
dependencies also make real FastAPI tests mandatory in `just ci`; they cover a
successful solve and child span, validation, auth and rate-limit short-circuits,
and middleware ordering.

## Live smoke query

After deploying with the secret attached:

1. Send a request that carries a known trace id, so you can find it without
   guessing:

   ```bash
   TRACE_ID=$(python3 -c "import secrets; print(secrets.token_hex(16))")
   echo "$TRACE_ID"
   curl -sS -o /dev/null -w '%{http_code}\n' \
     -H "traceparent: 00-${TRACE_ID}-00f067aa0ba902b7-01" \
     https://evan058--linprogx-demo-fastapi-app.modal.run/api/health
   ```

2. Wait a few seconds for the background exporter and Grafana ingestion, then
   look the trace up. From the site repo:

   ```bash
   site-traces trace "$TRACE_ID" --json
   ```

   or in Grafana Explore, TraceQL:

   ```
   { resource.service.name = "linprogx" }
   ```

3. Confirm: the `GET /api/health` span exists under `service.name=linprogx`, its
   `parentSpanId` is `00f067aa0ba902b7`, and `faas.coldstart` is `true` on the
   first request after a scale-from-zero and `false` on an immediate repeat.

4. Exercise the real path end to end from a browser on evanoman.com and confirm
   the browser fetch, `evanoman-linprogx-proxy`, and `linprogx` spans all share
   one trace id, with `solve.network_flow` beneath.

To verify the fail-open property in production, point
`OTEL_EXPORTER_OTLP_ENDPOINT` at an unreachable host and confirm `/api/health`
and `/api/solve/network-flow` still answer normally.
