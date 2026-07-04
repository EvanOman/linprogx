import { useProductionMix } from "./useProductionMix";
import { ResourceSlider } from "./components/ResourceSlider";
import { ResultsDisplay } from "./components/ResultsDisplay";
import { SolverStatus } from "./components/SolverStatus";
import type { ProductInput, ResourceInput } from "./types";

const DEFAULT_PRODUCTS: ProductInput[] = [
  { name: "Chairs", profit: 45 },
  { name: "Tables", profit: 80 },
  { name: "Bookcases", profit: 65 },
];

const DEFAULT_RESOURCES: ResourceInput[] = [
  { name: "Wood (board ft)", capacity: 400, usage: [5, 20, 10] },
  { name: "Labor (hours)", capacity: 450, usage: [10, 15, 12] },
  { name: "Finishing (hours)", capacity: 200, usage: [4, 8, 6] },
  { name: "Storage (sq ft)", capacity: 300, usage: [2, 8, 6] },
];

const DEFAULT_CAPACITIES = DEFAULT_RESOURCES.map((r) => r.capacity);

interface ProductionMixDemoProps {
  apiUrl: string;
}

export function ProductionMixDemo({ apiUrl }: ProductionMixDemoProps) {
  const { resources, setResourceCapacity, result, status, error, retry } =
    useProductionMix({
      apiUrl,
      products: DEFAULT_PRODUCTS,
      initialResources: DEFAULT_RESOURCES,
    });

  return (
    <div className="w-full max-w-5xl mx-auto px-4 py-8 space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl sm:text-4xl font-display font-bold text-fg">
          Production Mix Optimizer
        </h1>
        <p className="text-fg-muted max-w-2xl mx-auto">
          Adjust resource capacities to find the profit-maximizing product mix.
          Solves run live on{" "}
          <a
            href="https://github.com/EvanOman/linprogx"
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent-secondary hover:text-accent transition-colors"
          >
            linprogx
          </a>
          , a from-scratch LP solver.
        </p>
      </div>

      {/* Main content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left column: Controls */}
        <div className="space-y-6">
          {/* Products summary card */}
          <div className="rounded-card border border-surface-border bg-surface/30 backdrop-blur-sm p-5">
            <h2 className="text-sm font-medium text-fg-muted uppercase tracking-wider mb-4">
              Products
            </h2>
            <div className="grid grid-cols-3 gap-3">
              {DEFAULT_PRODUCTS.map((product) => (
                <div
                  key={product.name}
                  className="rounded-lg bg-surface-elevated/50 border border-surface-border p-3 text-center"
                >
                  <div className="text-sm font-medium text-fg">
                    {product.name}
                  </div>
                  <div className="text-lg font-mono font-bold text-accent mt-1">
                    ${product.profit}
                  </div>
                  <div className="text-[10px] text-fg-faint uppercase tracking-wider mt-0.5">
                    profit/unit
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Resource sliders card */}
          <div className="rounded-card border border-surface-border bg-surface/30 backdrop-blur-sm p-5">
            <h2 className="text-sm font-medium text-fg-muted uppercase tracking-wider mb-4">
              Resource Capacities
            </h2>
            <div className="space-y-5">
              {resources.map((resource, i) => (
                <ResourceSlider
                  key={resource.name}
                  resource={resource}
                  index={i}
                  defaultCapacity={DEFAULT_CAPACITIES[i]}
                  onChange={setResourceCapacity}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Right column: Results */}
        <div className="space-y-6">
          <div className="rounded-card border border-surface-border bg-surface/30 backdrop-blur-sm p-5">
            <h2 className="text-sm font-medium text-fg-muted uppercase tracking-wider mb-4">
              Optimal Solution
            </h2>
            {(status === "idle" ||
              status === "loading" ||
              status === "cold-start") &&
              !result && (
                <div className="py-12 text-center text-fg-faint">
                  {status === "cold-start"
                    ? "Waiting for solver to wake up..."
                    : "Computing optimal solution..."}
                </div>
              )}
            {result && <ResultsDisplay result={result} />}
            {status === "error" && !result && (
              <div className="py-12 text-center text-fg-faint">
                No solution available. Check the error below.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Solver status footer */}
      <SolverStatus
        status={status}
        result={result}
        error={error}
        onRetry={retry}
      />

      {/* Resource usage table */}
      <div className="rounded-card border border-surface-border bg-surface/30 backdrop-blur-sm p-5">
        <h2 className="text-sm font-medium text-fg-muted uppercase tracking-wider mb-4">
          Resource Usage per Product
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-surface-border">
                <th className="text-left py-2 pr-4 text-fg-muted font-medium">
                  Resource
                </th>
                {DEFAULT_PRODUCTS.map((p) => (
                  <th
                    key={p.name}
                    className="text-right py-2 px-3 text-fg-muted font-medium"
                  >
                    {p.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {DEFAULT_RESOURCES.map((resource) => (
                <tr
                  key={resource.name}
                  className="border-b border-surface-border/50"
                >
                  <td className="py-2 pr-4 text-fg">{resource.name}</td>
                  {resource.usage.map((u, i) => (
                    <td
                      key={DEFAULT_PRODUCTS[i].name}
                      className="text-right py-2 px-3 font-mono text-fg-muted"
                    >
                      {u}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
