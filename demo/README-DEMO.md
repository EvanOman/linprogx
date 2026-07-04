# Production Mix Optimizer -- linprogx Demo

Interactive LP demo for evanoman.com: users adjust factory resource capacities
with sliders and see the profit-maximizing production plan update in real time,
solved by linprogx.

## Component

**Main export:** `ProductionMixDemo` from `demo/web/src/ProductionMixDemo.tsx`

```tsx
interface ProductionMixDemoProps {
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
| `src/ProductionMixDemo.tsx` | Main component -- products summary, resource sliders, results grid |
| `src/useProductionMix.ts` | Custom hook -- state, 300ms debounced API calls, abort controller, cold-start detection |
| `src/types.ts` | TypeScript types for the API contract |
| `src/components/ResourceSlider.tsx` | Range slider with gradient fill and reset button |
| `src/components/ResultsDisplay.tsx` | Bars for quantities, utilization, shadow price tooltips |
| `src/components/SolverStatus.tsx` | Loading/cold-start/error/solved states with engine credit |
| `src/index.css` | Tailwind directives + custom slider thumb styling |

### Dev server

```bash
cd demo/web
npm install
npm run dev    # starts on http://localhost:19100
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
  "demo": "production-mix-optimizer"
}
```

### `GET /api/health`

Returns `{"status": "ok"}`.

### `POST /api/solve/production-mix`

Structured endpoint for the production-mix LP problem.

**Request:**
```json
{
  "products": [
    {"name": "Chairs", "profit": 45},
    {"name": "Tables", "profit": 80}
  ],
  "resources": [
    {"name": "Wood", "capacity": 400, "usage": [5, 20]},
    {"name": "Labor", "capacity": 450, "usage": [10, 15]}
  ]
}
```

**Response:**
```json
{
  "status": "optimal",
  "total_profit": 2200.0,
  "products": [
    {"name": "Chairs", "quantity": 24.0, "profit_contribution": 1080.0},
    {"name": "Tables", "quantity": 14.0, "profit_contribution": 1120.0}
  ],
  "resources": [
    {
      "name": "Wood",
      "used": 400.0,
      "capacity": 400.0,
      "utilization": 1.0,
      "shadow_price": 1.0,
      "binding": true
    }
  ],
  "iterations": 2,
  "solve_time_ms": 0.08,
  "solver": "linprogx v0.1.0"
}
```

**Constraints:**
- Max 10 products, max 10 resources
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
