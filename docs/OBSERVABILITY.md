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
  instrumentation adds hundreds of milliseconds of import time to every cold
  start — the exact number this instrumentation exists to measure;
- the edge exporter (`functions/_observability.ts` in the site repo) is already
  a hand-rolled OTLP/HTTP JSON client, so matching it keeps one contract;
- it adds no dependency, no lockfile churn, and no package release-age exposure.

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

Only `http://` and `https://` endpoints are accepted; anything else is treated
as unconfigured.

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

| Span | Kind | When |
| --- | --- | --- |
| `{METHOD} {route}` | server | Every HTTP request. Route is one of `/api/health`, `/api/info`, `/api/solve/network-flow`, or `/*`. |
| `container.init` | internal | First request in a process only; covers interpreter/import time. |
| `solve.network_flow` | internal | The min-cost-flow LP run. |

Notable attributes:

- request: `http.request.method`, `http.route`, `http.response.status_code`,
  `request.outcome` (`success` / `not_found` / `invalid_request` /
  `rate_limited` / `timeout` / `client_error` / `internal_error` /
  `cancelled`),
  `request.correlation_id` (only if it matches a UUID/trace-id shape),
  `request.purpose=warmup`, `faas.coldstart`, `faas.instance`,
  `container.reused`, and bounded `container.reuse_bucket`
  (`first` / `2-5` / `6-20` / `21+`).
- solve: `solve.result` (`optimal` / `infeasible` / `unbounded` /
  `iteration_limit` / `timeout` / `error`), `lp.nodes`, `lp.edges`,
  `lp.iterations`, `lp.solve_time_ms`.

`solve.result` is bounded by construction: `linprogx.Status` is a four-value
`StrEnum`, plus the two outcomes the endpoint adds itself.

Only 5xx sets span status `ERROR`. Infeasible or unbounded results, a hit solve
cap, the 30 req/min rate limit, 4xx validation failures, and a client that
disconnects mid-request (`request.outcome=cancelled`) are recorded as outcomes,
not incidents — they are the API working correctly.

### Cold starts

`container.init` deliberately starts *before* its parent request span: on Modal's
Linux runtime it uses the process start clock to cover interpreter startup,
imports, and the short pre-request interval. Attaching it to the first request is
what makes a cold start visible in the same waterfall as the latency it caused.
Non-Linux local development falls back to the tracing module's import time.

Modal memory-snapshot restores resume a process whose monotonic clock predates
the snapshot, which would make the measured init window meaningless. When the
measurement exceeds 120 s it is discarded and only `faas.coldstart=true` is
reported. Container identity is likewise minted lazily from runtime process
identity rather than at import, so one snapshotted id is not cloned into every
restored container.

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
- connection failures, timeouts, and non-2xx responses are swallowed and counted,
  logged at most once a minute, and never include the URL, credentials, or the
  exception;
- with no endpoint configured, the whole path costs a few dict lookups.

The exporter thread starts on first use rather than at import, keeping it out of
Modal's memory snapshot; a pid/liveness check restarts it after a snapshot
restore or fork.

`tests/test_demo_tracing.py` covers all of this — valid, invalid, and missing
trace context, unsampled upstream decisions, collector connection failure,
collector 500, a hanging collector, unconfigured and disabled exporters, and the
privacy assertions. It drives the middleware through the raw ASGI protocol, so
the tracing contract is tested without adding a web test stack to a solver repo.
The two tests that exercise the real FastAPI app skip where FastAPI is absent.

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

2. Wait a few seconds (the exporter batches on a 250 ms tick; Grafana ingestion
   adds a little more), then look the trace up. From the site repo:

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
