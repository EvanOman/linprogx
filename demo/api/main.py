"""linprogx demo API — Production Mix Optimizer.

A purpose-built FastAPI service that solves production-mix LP problems
using linprogx.  Deployed on Render for the evanoman.com interactive demo.
"""

from __future__ import annotations

import signal
import time
from contextlib import contextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import linprogx

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
# CORS — restricted to evanoman.com + local dev
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS = [
    "https://evanoman.com",
    "https://www.evanoman.com",
    "http://localhost:5173",
    "http://localhost:5174",
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
# Rate limiting — simple in-memory sliding window
# ---------------------------------------------------------------------------

_RATE_WINDOW = 60  # seconds
_RATE_LIMIT = 30  # requests per window per IP

_rate_buckets: dict[str, list[float]] = {}


def _check_rate_limit(client_ip: str) -> bool:
    now = time.monotonic()
    bucket = _rate_buckets.setdefault(client_ip, [])
    # Prune old entries
    bucket[:] = [t for t in bucket if now - t < _RATE_WINDOW]
    if len(bucket) >= _RATE_LIMIT:
        return False
    bucket.append(now)
    return True


# Periodic cleanup to avoid memory leak from stale IPs
_last_cleanup = time.monotonic()


def _maybe_cleanup() -> None:
    global _last_cleanup
    now = time.monotonic()
    if now - _last_cleanup < 300:  # every 5 min
        return
    _last_cleanup = now
    stale = [
        ip
        for ip, times in _rate_buckets.items()
        if not times or now - times[-1] > _RATE_WINDOW
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


# ---------------------------------------------------------------------------
# Solve timeout
# ---------------------------------------------------------------------------

SOLVE_TIMEOUT_SECONDS = 5


class SolveTimeoutError(Exception):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise SolveTimeoutError()


@contextmanager
def solve_timeout(seconds: int = SOLVE_TIMEOUT_SECONDS):
    """Context manager that raises SolveTimeoutError after *seconds*."""
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# Hard caps
# ---------------------------------------------------------------------------

MAX_PRODUCTS = 10
MAX_RESOURCES = 10
MAX_COEFFICIENT = 1_000_000


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class Product(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    profit: float = Field(..., ge=0, le=MAX_COEFFICIENT)

    @field_validator("profit")
    @classmethod
    def profit_finite(cls, v: float) -> float:
        if not (-MAX_COEFFICIENT <= v <= MAX_COEFFICIENT):
            raise ValueError(f"profit must be between -{MAX_COEFFICIENT} and {MAX_COEFFICIENT}")
        return v


class Resource(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    capacity: float = Field(..., gt=0, le=MAX_COEFFICIENT)
    usage: list[float] = Field(..., min_length=1, max_length=MAX_PRODUCTS)

    @field_validator("usage")
    @classmethod
    def usage_nonneg(cls, v: list[float]) -> list[float]:
        for u in v:
            if u < 0 or u > MAX_COEFFICIENT:
                raise ValueError(f"usage values must be between 0 and {MAX_COEFFICIENT}")
        return v


class SolveRequest(BaseModel):
    products: list[Product] = Field(..., min_length=1, max_length=MAX_PRODUCTS)
    resources: list[Resource] = Field(..., min_length=1, max_length=MAX_RESOURCES)


class ProductResult(BaseModel):
    name: str
    quantity: float
    profit_contribution: float


class ResourceResult(BaseModel):
    name: str
    used: float
    capacity: float
    utilization: float  # 0..1
    shadow_price: float
    binding: bool


class SolveResponse(BaseModel):
    status: str
    total_profit: float | None = None
    products: list[ProductResult] = []
    resources: list[ResourceResult] = []
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
    demo: str = "production-mix-optimizer"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/info", response_model=InfoResponse)
def solver_info() -> InfoResponse:
    return InfoResponse()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/solve/production-mix", response_model=SolveResponse)
def solve_production_mix(req: SolveRequest) -> SolveResponse:
    n_products = len(req.products)
    n_resources = len(req.resources)

    # Validate usage vector lengths
    for r in req.resources:
        if len(r.usage) != n_products:
            raise HTTPException(
                status_code=422,
                detail=f"Resource '{r.name}' usage list has {len(r.usage)} entries "
                f"but there are {n_products} products",
            )

    # Build the LP:  maximize profit^T x  s.t.  usage * x <= capacity, x >= 0
    c = [p.profit for p in req.products]

    model = linprogx.Model(name="production_mix")
    variables = [model.variable(name=p.name, lower=0.0) for p in req.products]

    model.maximize({v: p.profit for v, p in zip(variables, req.products)})

    for resource in req.resources:
        model.add_constraint(
            {v: u for v, u in zip(variables, resource.usage)},
            "<=",
            resource.capacity,
            name=resource.name,
        )

    t0 = time.perf_counter()
    try:
        with solve_timeout():
            solution = model.solve()
    except SolveTimeoutError:
        return SolveResponse(
            status="timeout",
            solve_time_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Solve failed: {exc}") from exc
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    if solution.status != linprogx.Status.OPTIMAL:
        return SolveResponse(
            status=str(solution.status),
            iterations=solution.iterations,
            solve_time_ms=elapsed_ms,
        )

    # Build response
    product_results = []
    for i, p in enumerate(req.products):
        qty = solution.x[i] if i < len(solution.x) else 0.0
        product_results.append(
            ProductResult(
                name=p.name,
                quantity=round(qty, 6),
                profit_contribution=round(qty * p.profit, 6),
            )
        )

    resource_results = []
    sensitivity = solution.sensitivity
    for i, r in enumerate(req.resources):
        used = sum(
            r.usage[j] * (solution.x[j] if j < len(solution.x) else 0.0)
            for j in range(n_products)
        )
        shadow = 0.0
        if sensitivity and i < len(sensitivity.shadow_prices):
            shadow = sensitivity.shadow_prices[i]
        utilization = used / r.capacity if r.capacity > 0 else 0.0
        resource_results.append(
            ResourceResult(
                name=r.name,
                used=round(used, 6),
                capacity=r.capacity,
                utilization=round(min(utilization, 1.0), 6),
                shadow_price=round(shadow, 6),
                binding=utilization > 0.999,
            )
        )

    return SolveResponse(
        status="optimal",
        total_profit=round(solution.objective_value, 6) if solution.objective_value is not None else None,
        products=product_results,
        resources=resource_results,
        iterations=solution.iterations,
        solve_time_ms=elapsed_ms,
    )
