import type { SolveResponse } from "../types";

interface ResultsDisplayProps {
  result: SolveResponse;
}

function formatCurrency(value: number): string {
  return `$${value.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  })}`;
}

function formatQuantity(value: number): string {
  return value % 1 === 0
    ? value.toLocaleString()
    : value.toLocaleString(undefined, {
        minimumFractionDigits: 1,
        maximumFractionDigits: 2,
      });
}

function utilizationColor(utilization: number, binding: boolean): string {
  if (binding) return "bg-accent";
  if (utilization > 0.8) return "bg-amber-500";
  return "bg-accent-secondary";
}

function utilizationTextColor(utilization: number, binding: boolean): string {
  if (binding) return "text-accent";
  if (utilization > 0.8) return "text-amber-500";
  return "text-accent-secondary";
}

function ShadowPriceLabel({
  shadowPrice,
  resourceName,
  binding,
}: {
  shadowPrice: number;
  resourceName: string;
  binding: boolean;
}) {
  if (shadowPrice === 0) {
    return (
      <span className="text-xs text-fg-faint">
        Marginal value: $0.00/unit (surplus available)
      </span>
    );
  }

  return (
    <div className="group/tip relative inline-flex items-center gap-1">
      <span
        className={`text-xs font-mono font-medium ${
          binding ? "text-accent" : "text-fg-muted"
        }`}
      >
        Marginal value: {formatCurrency(shadowPrice)}/unit
      </span>
      <span className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full border border-surface-border text-fg-faint text-[10px] cursor-help">
        ?
      </span>
      <div className="absolute bottom-full left-0 mb-2 w-64 p-2.5 rounded-lg bg-surface-elevated border border-surface-border text-xs text-fg-muted opacity-0 pointer-events-none group-hover/tip:opacity-100 group-hover/tip:pointer-events-auto transition-opacity z-10 shadow-lg">
        Each additional unit of {resourceName} would increase total profit by{" "}
        {formatCurrency(shadowPrice)}. This constraint is limiting your
        production.
      </div>
    </div>
  );
}

export function ResultsDisplay({ result }: ResultsDisplayProps) {
  const maxQuantity = Math.max(...result.products.map((p) => p.quantity), 1);
  const maxContribution = Math.max(
    ...result.products.map((p) => p.profit_contribution),
    1
  );

  return (
    <div className="space-y-6">
      {/* Total Profit */}
      <div className="text-center py-4 rounded-card bg-surface/50 border border-surface-border">
        <div className="text-sm text-fg-muted uppercase tracking-wider mb-1">
          Optimal Total Profit
        </div>
        <div className="text-4xl font-display font-bold text-accent">
          {formatCurrency(result.total_profit)}
        </div>
      </div>

      {/* Product Quantities */}
      <div>
        <h3 className="text-sm font-medium text-fg-muted uppercase tracking-wider mb-3">
          Optimal Production
        </h3>
        <div className="space-y-3">
          {result.products.map((product) => {
            const qPct =
              maxQuantity > 0 ? (product.quantity / maxQuantity) * 100 : 0;
            const cPct =
              maxContribution > 0
                ? (product.profit_contribution / maxContribution) * 100
                : 0;

            return (
              <div key={product.name} className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-fg">
                    {product.name}
                  </span>
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-mono text-fg-muted">
                      {formatQuantity(product.quantity)} units
                    </span>
                    <span className="text-sm font-mono font-medium text-accent">
                      {formatCurrency(product.profit_contribution)}
                    </span>
                  </div>
                </div>
                <div className="flex gap-1 h-2">
                  <div className="flex-1 rounded-full bg-surface-elevated overflow-hidden">
                    <div
                      className="h-full rounded-full bg-accent-secondary/70 transition-all duration-500 ease-out"
                      style={{ width: `${qPct}%` }}
                      title={`${formatQuantity(product.quantity)} units`}
                    />
                  </div>
                  <div className="flex-1 rounded-full bg-surface-elevated overflow-hidden">
                    <div
                      className="h-full rounded-full bg-accent/70 transition-all duration-500 ease-out"
                      style={{ width: `${cPct}%` }}
                      title={`${formatCurrency(product.profit_contribution)} profit`}
                    />
                  </div>
                </div>
                <div className="flex gap-1 text-[10px] text-fg-faint">
                  <span className="flex-1">Quantity</span>
                  <span className="flex-1">Profit contribution</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Resource Utilization */}
      <div>
        <h3 className="text-sm font-medium text-fg-muted uppercase tracking-wider mb-3">
          Resource Utilization
        </h3>
        <div className="space-y-4">
          {result.resources.map((resource) => {
            const pct = resource.utilization * 100;
            const barColor = utilizationColor(
              resource.utilization,
              resource.binding
            );
            const textColor = utilizationTextColor(
              resource.utilization,
              resource.binding
            );

            return (
              <div key={resource.name} className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-fg">
                    {resource.name}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-mono text-fg-muted">
                      {formatQuantity(resource.used)} /{" "}
                      {formatQuantity(resource.capacity)}
                    </span>
                    {resource.binding && (
                      <span className="text-[10px] font-medium uppercase tracking-wider px-1.5 py-0.5 rounded bg-accent/10 text-accent border border-accent/20">
                        Binding
                      </span>
                    )}
                  </div>
                </div>
                <div className="h-3 rounded-full bg-surface-elevated overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ease-out ${barColor}`}
                    style={{ width: `${Math.min(pct, 100)}%` }}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <ShadowPriceLabel
                    shadowPrice={resource.shadow_price}
                    resourceName={resource.name}
                    binding={resource.binding}
                  />
                  <span className={`text-xs font-mono ${textColor}`}>
                    {pct.toFixed(1)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
