# Network Flow Optimizer -- linprogx Demo

Interactive supply-chain network flow demo for evanoman.com: users adjust
supply capacities, demand requirements, and route capacities with sliders and
watch the optimal flows reroute live through an animated SVG network diagram,
solved by linprogx.

## Component

**Main export:** `NetworkFlowDemo` from `demo/web/src/NetworkFlowDemo.tsx`

```tsx
interface NetworkFlowDemoProps {
  apiUrl: string;  // e.g. "https://linprogx-demo.onrender.com"
}
```

### Integration into evanoman.com

The component is self-contained React + TypeScript + Tailwind. To embed it:

1. Copy `demo/web/src/` into the website's source tree (or import as a route).
2. The Tailwind config in `demo/web/tailwind.config.ts` mirrors the website's
   design tokens (colors, fonts, border radius). When integrating, the
   component will pick up the website's existing theme -- the class names
   (`text-fg`, `bg-surface`, `text-accent`, etc.) are intentionally identical.
3. Pass `apiUrl="https://linprogx-demo.onrender.com"` as a prop.
4. The component handles its own state, debounced API calls, cold-start UX,
   error handling, and retry logic.

### Key files

| File | Purpose |
|------|---------|
| `src/NetworkFlowDemo.tsx` | Main component -- SVG graph, controls, cost summary, flow table |
| `src/useNetworkFlow.ts` | Custom hook -- state, 300ms debounced API calls, abort controller, cold-start detection |
| `src/config.ts` | Default network (7 nodes, 9 edges with positions, costs, capacities) |
| `src/types.ts` | TypeScript types for the API contract and graph layout |
| `src/components/NetworkGraph.tsx` | Animated SVG network diagram with flow particles, edge utilization colors, hover tooltips |
| `src/components/ControlPanel.tsx` | Sliders for supply, demand, and edge capacities with reset buttons |
| `src/components/SolverStatus.tsx` | Loading/cold-start/error/solved states with engine credit |
| `src/components/AnalogsBanner.tsx` | Note about analogous LP applications |
| `src/index.css` | Tailwind directives + custom slider thumb styling |

### Default network

A supply-chain logistics problem:

- **Supply nodes:** Seattle (400 units), Houston (500 units)
- **Hub nodes:** Denver, Atlanta (flow conservation)
- **Demand nodes:** New York (300), Chicago (250), Miami (200)
- **9 edges** with per-unit shipping costs ($2-$8) and capacity limits

The LP minimizes total shipping cost subject to supply limits, demand
requirements, flow conservation at hubs, and edge capacities.

### Dev server

```bash
cd demo/web
npm install
npm run dev    # starts on http://localhost:19101
```

Set `VITE_API_URL` to override the API endpoint (defaults to the deployed Render URL).

## API

**Base URL:** `https://linprogx-demo.onrender.com`

### `GET /api/info`

Returns solver metadata:
```json
{
  "solver": "linprogx",
  "version": "0.1.0",
  "description": "A from-scratch LP solver with two-phase simplex...",
  "github": "https://github.com/EvanOman/linprogx",
  "demo": "network-flow-optimizer"
}
```

### `GET /api/health`

Returns `{"status": "ok"}`.

### `POST /api/solve/network-flow`

Structured endpoint for the min-cost network flow LP.

**Request:**
```json
{
  "nodes": [
    {"id": "seattle", "type": "supply", "value": 400},
    {"id": "denver", "type": "hub", "value": 0},
    {"id": "nyc", "type": "demand", "value": 300}
  ],
  "edges": [
    {"from": "seattle", "to": "denver", "cost": 5, "capacity": 300}
  ]
}
```

**Response:**
```json
{
  "status": "optimal",
  "total_cost": 5900.0,
  "flows": [
    {
      "from": "seattle",
      "to": "denver",
      "flow": 50.0,
      "capacity": 300.0,
      "utilization": 0.1667,
      "cost": 5.0,
      "flow_cost": 250.0
    }
  ],
  "node_balances": [
    {"id": "seattle", "type": "supply", "value": 400, "net_flow": 250.0}
  ],
  "iterations": 11,
  "solve_time_ms": 0.77,
  "solver": "linprogx v0.1.0"
}
```

**Constraints:**
- Max 20 nodes, max 50 edges
- Coefficient values capped at 1,000,000
- Solve timeout: 5 seconds
- Rate limit: 30 requests/minute/IP

**CORS:** Restricted to `evanoman.com`, `www.evanoman.com`, and localhost dev ports.

## Deployment

The API runs on Render (free tier, Oregon). Cold start after inactivity takes
30-60 seconds -- the frontend shows a "Waking up the solver" state during this.

**Render service:** `linprogx-demo` (Python runtime)
**Build command:** `chmod +x render-build.sh && ./render-build.sh`
**Start command:** `uvicorn demo.api.main:app --host 0.0.0.0 --port $PORT`

The C extensions build at deploy time. OpenBLAS is detected automatically;
the solver works with or without it (BLAS only matters for large sparse problems).

## Observability

The API continues the W3C trace context the Cloudflare Pages proxy forwards, so
browser, edge, and Modal spans land in one trace. Configuration is two
environment variables (`OTEL_EXPORTER_OTLP_ENDPOINT` and
`OTEL_EXPORTER_OTLP_AUTHORIZATION`), supplied on Modal by the `otel-grafana`
secret. Tracing is fail-open and exports nothing when unconfigured. See
[docs/OBSERVABILITY.md](../docs/OBSERVABILITY.md).
