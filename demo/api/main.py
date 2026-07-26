"""linprogx demo API — Network Flow Optimizer.

A purpose-built FastAPI service that solves minimum-cost network-flow LP
problems using linprogx.  Deployed on Render for the evanoman.com
interactive demo.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import linprogx
from demo.api import tracing

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="linprogx Demo API",
    version=linprogx.__version__,
    docs_url=None,
    redoc_url=None,
)

# ---------------------------------------------------------------------------
# Tracing — continues the browser -> Cloudflare -> Modal trace. Fail-open: with
# no OTLP endpoint configured this costs a few dict lookups per request and
# exports nothing. See docs/OBSERVABILITY.md.
# ---------------------------------------------------------------------------

_SERVICE_NAME = "linprogx"
# Bounds http.route cardinality; anything else is reported as "/*".
_TRACED_ROUTES = frozenset({"/api/health", "/api/info", "/api/solve/network-flow"})
tracing.configure(_SERVICE_NAME)

# ---------------------------------------------------------------------------
# CORS — restricted to evanoman.com + local dev
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS = [
    "https://evanoman.com",
    "https://www.evanoman.com",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:14173",
    "http://localhost:18920",
    "http://localhost:19100",
    "http://localhost:19101",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=3600,
)

# ---------------------------------------------------------------------------
# Auth — require X-Demo-Secret header on /api/* (except /api/health)
# ---------------------------------------------------------------------------

_DEMO_SECRET = os.environ.get("DEMO_SHARED_SECRET", "")


@app.middleware("http")
async def auth_middleware(request: Request, call_next: Any) -> Response:
    path = request.url.path
    if _DEMO_SECRET and path.startswith("/api/") and path != "/api/health":
        provided = request.headers.get("X-Demo-Secret", "")
        if provided != _DEMO_SECRET:
            return Response(
                content='{"detail":"Unauthorized"}',
                status_code=401,
                media_type="application/json",
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Rate limiting — simple in-memory sliding window
# ---------------------------------------------------------------------------

_RATE_WINDOW = 60  # seconds
_RATE_LIMIT = 30  # requests per window per IP

_rate_buckets: dict[str, list[float]] = {}


def _check_rate_limit(client_ip: str) -> bool:
    now = time.monotonic()
    bucket = _rate_buckets.setdefault(client_ip, [])
    bucket[:] = [t for t in bucket if now - t < _RATE_WINDOW]
    if len(bucket) >= _RATE_LIMIT:
        return False
    bucket.append(now)
    return True


_last_cleanup = time.monotonic()


def _maybe_cleanup() -> None:
    global _last_cleanup
    now = time.monotonic()
    if now - _last_cleanup < 300:
        return
    _last_cleanup = now
    stale = [
        ip for ip, times in _rate_buckets.items() if not times or now - times[-1] > _RATE_WINDOW
    ]
    for ip in stale:
        del _rate_buckets[ip]


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next: Any) -> Response:
    _maybe_cleanup()
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        return Response(
            content='{"detail":"Rate limit exceeded. Try again in a minute."}',
            status_code=429,
            media_type="application/json",
        )
    response = await call_next(request)
    return response


# Added after the auth and rate-limit middleware so it ends up outermost: the
# server span then covers the whole request, including the 401s and 429s those
# two short-circuit, and records them as outcomes rather than losing them.
app.add_middleware(tracing.TracingMiddleware, routes=_TRACED_ROUTES)


# ---------------------------------------------------------------------------
# Solve timeout & thread pool
# ---------------------------------------------------------------------------

SOLVE_TIMEOUT_SECONDS = 5
_solve_pool = ThreadPoolExecutor(max_workers=2)

# ---------------------------------------------------------------------------
# Hard caps
# ---------------------------------------------------------------------------

MAX_NODES = 20
MAX_EDGES = 50
MAX_VALUE = 100_000


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class Node(BaseModel):
    id: str = Field(..., min_length=1, max_length=30)
    type: Literal["supply", "hub", "demand"]
    value: float = Field(default=0, ge=0, le=MAX_VALUE)

    @field_validator("value")
    @classmethod
    def value_required_for_supply_demand(cls, v: float, info: Any) -> float:
        # supply and demand nodes need a positive value; hubs default to 0
        return v


class Edge(BaseModel):
    source: str = Field(..., min_length=1, max_length=30, alias="from")
    target: str = Field(..., min_length=1, max_length=30, alias="to")
    cost: float = Field(..., ge=0, le=MAX_VALUE)
    capacity: float = Field(..., gt=0, le=MAX_VALUE)

    model_config = {"populate_by_name": True}


class FlowRequest(BaseModel):
    nodes: list[Node] = Field(..., min_length=2, max_length=MAX_NODES)
    edges: list[Edge] = Field(..., min_length=1, max_length=MAX_EDGES)


class FlowResult(BaseModel):
    source: str = Field(alias="from")
    target: str = Field(alias="to")
    flow: float
    capacity: float
    utilization: float
    cost: float
    flow_cost: float

    model_config = {"populate_by_name": True}


class NodeBalance(BaseModel):
    id: str
    type: str
    value: float
    net_flow: float


class FlowResponse(BaseModel):
    status: str
    total_cost: float | None = None
    flows: list[FlowResult] = []
    node_balances: list[NodeBalance] = []
    iterations: int = 0
    solve_time_ms: float = 0
    solver: str = f"linprogx v{linprogx.__version__}"


class InfoResponse(BaseModel):
    solver: str = "linprogx"
    version: str = linprogx.__version__
    description: str = (
        "A from-scratch LP solver with two-phase simplex, interior-point, "
        "and PDHG methods. Competitive with HiGHS and Clarabel on the "
        "Netlib benchmark suite."
    )
    github: str = "https://github.com/EvanOman/linprogx"
    demo: str = "network-flow-optimizer"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/info", response_model=InfoResponse)
def solver_info() -> InfoResponse:
    return InfoResponse()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/solve/network-flow", response_model=FlowResponse)
def solve_network_flow(req: FlowRequest) -> FlowResponse:
    node_ids = {n.id for n in req.nodes}

    # Validate edges reference valid nodes
    for e in req.edges:
        if e.source not in node_ids:
            raise HTTPException(422, f"Edge references unknown source node '{e.source}'")
        if e.target not in node_ids:
            raise HTTPException(422, f"Edge references unknown target node '{e.target}'")

    # Build the min-cost flow LP:
    #   minimize  sum(cost_e * x_e)
    #   subject to:
    #     For supply nodes:  sum(outflow) - sum(inflow) <= supply
    #     For demand nodes:  sum(inflow) - sum(outflow) >= demand
    #     For hub nodes:     sum(inflow) = sum(outflow)  (flow conservation)
    #     For all edges:     0 <= x_e <= capacity_e
    model = linprogx.Model(name="network_flow")

    # Create a flow variable for each edge
    edge_vars = []
    for e in req.edges:
        v = model.variable(name=f"{e.source}->{e.target}", lower=0.0, upper=e.capacity)
        edge_vars.append(v)

    # Objective: minimize total shipping cost
    model.minimize({v: e.cost for v, e in zip(edge_vars, req.edges, strict=True)})

    # Flow balance constraints for each node
    for node in req.nodes:
        # Coefficients: +1 for outgoing edges, -1 for incoming edges
        coeffs: dict[Any, float] = {}
        for v, e in zip(edge_vars, req.edges, strict=True):
            if e.source == node.id:
                coeffs[v] = coeffs.get(v, 0.0) + 1.0  # outflow
            if e.target == node.id:
                coeffs[v] = coeffs.get(v, 0.0) - 1.0  # inflow (negative)

        if not coeffs:
            continue

        # net_outflow = sum(outflow) - sum(inflow)
        if node.type == "supply":
            # net_outflow <= supply (can ship at most supply amount)
            model.add_constraint(coeffs, "<=", node.value, name=f"supply_{node.id}")
        elif node.type == "demand":
            # net_outflow <= -demand  =>  net_inflow >= demand
            model.add_constraint(coeffs, "<=", -node.value, name=f"demand_{node.id}")
        else:
            # hub: flow conservation (net_outflow = 0)
            model.add_constraint(coeffs, "=", 0.0, name=f"balance_{node.id}")

    # The LP itself gets its own span: shape and outcome only, never node or edge
    # names. `solve.result` is a bounded vocabulary -- linprogx.Status is a
    # four-value StrEnum, plus the two outcomes this endpoint adds itself.
    solve_span = tracing.get_tracer().start_span(
        "solve.network_flow",
        attributes={"lp.nodes": len(req.nodes), "lp.edges": len(req.edges)},
    )
    t0 = time.perf_counter()
    try:
        future = _solve_pool.submit(model.solve)
        solution = future.result(timeout=SOLVE_TIMEOUT_SECONDS)
    except FuturesTimeoutError:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        # A hit solve cap is an expected outcome at demo scale, not an incident:
        # the span status stays UNSET so it never fires a 5xx-shaped alert.
        solve_span.set_attributes({"solve.result": "timeout", "lp.solve_time_ms": elapsed_ms})
        solve_span.end()
        return FlowResponse(status="timeout", solve_time_ms=elapsed_ms)
    except Exception as exc:
        solve_span.record_exception(exc)
        solve_span.set_status(tracing.STATUS_ERROR)
        solve_span.set_attribute("solve.result", "error")
        solve_span.end()
        raise HTTPException(status_code=500, detail=f"Solve failed: {exc}") from exc
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    solve_span.set_attributes(
        {
            "solve.result": str(solution.status),
            "lp.iterations": solution.iterations,
            "lp.solve_time_ms": elapsed_ms,
        }
    )
    solve_span.end()

    if solution.status != linprogx.Status.OPTIMAL:
        return FlowResponse(
            status=str(solution.status),
            iterations=solution.iterations,
            solve_time_ms=elapsed_ms,
        )

    # Build flow results
    flow_results = []
    for i, e in enumerate(req.edges):
        flow_val = solution.x[i] if i < len(solution.x) else 0.0
        utilization = flow_val / e.capacity if e.capacity > 0 else 0.0
        flow_results.append(
            FlowResult(
                **{
                    "from": e.source,
                    "to": e.target,
                    "flow": round(flow_val, 4),
                    "capacity": e.capacity,
                    "utilization": round(min(utilization, 1.0), 4),
                    "cost": e.cost,
                    "flow_cost": round(flow_val * e.cost, 4),
                }
            )
        )

    # Compute node balances
    node_balances = []
    for node in req.nodes:
        net_out = 0.0
        for i, e in enumerate(req.edges):
            flow_val = solution.x[i] if i < len(solution.x) else 0.0
            if e.source == node.id:
                net_out += flow_val
            if e.target == node.id:
                net_out -= flow_val
        node_balances.append(
            NodeBalance(
                id=node.id,
                type=node.type,
                value=node.value,
                net_flow=round(abs(net_out), 4),
            )
        )

    total_cost = sum(fr.flow_cost for fr in flow_results)

    return FlowResponse(
        status="optimal",
        total_cost=round(total_cost, 4),
        flows=flow_results,
        node_balances=node_balances,
        iterations=solution.iterations,
        solve_time_ms=elapsed_ms,
    )
