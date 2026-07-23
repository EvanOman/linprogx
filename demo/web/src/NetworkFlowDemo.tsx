import { useNetworkFlow } from "./useNetworkFlow";
import { NetworkGraph } from "./components/NetworkGraph";
import { ControlPanel } from "./components/ControlPanel";
import { SolverStatus } from "./components/SolverStatus";
import { AnalogsBanner } from "./components/AnalogsBanner";
import { DEFAULT_NODES, DEFAULT_EDGES } from "./config";

interface NetworkFlowDemoProps {
  apiUrl: string;
}

export function NetworkFlowDemo({ apiUrl }: NetworkFlowDemoProps) {
  const {
    nodeValues,
    edgeCapacities,
    setNodeValue,
    setEdgeCapacity,
    resetAll,
    result,
    status,
    error,
    retry,
  } = useNetworkFlow({
    apiUrl,
    defaultNodes: DEFAULT_NODES,
    defaultEdges: DEFAULT_EDGES,
  });

  return (
    <div className="w-full max-w-6xl mx-auto px-4 py-8 space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl sm:text-4xl font-display font-bold text-fg">
          Network Flow Optimizer
        </h1>
        <p className="text-fg-muted max-w-2xl mx-auto">
          Adjust supply, demand, and route capacities to find the minimum-cost
          flow through the network. Solves run live on{" "}
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

      {/* Network graph -- the hero */}
      <div className="rounded-card border border-surface-border bg-surface/20 backdrop-blur-sm p-2 sm:p-4">
        <NetworkGraph
          nodes={DEFAULT_NODES}
          edges={DEFAULT_EDGES}
          nodeValues={nodeValues}
          edgeCapacities={edgeCapacities}
          flows={result?.flows ?? null}
        />
      </div>

      {/* Control panel + summary */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ControlPanel
            nodes={DEFAULT_NODES}
            edges={DEFAULT_EDGES}
            nodeValues={nodeValues}
            edgeCapacities={edgeCapacities}
            onNodeValueChange={setNodeValue}
            onEdgeCapacityChange={setEdgeCapacity}
            onResetAll={resetAll}
          />
        </div>

        {/* Summary strip */}
        <div className="space-y-4">
          {/* Total cost card */}
          <div className="rounded-card border border-surface-border bg-surface/30 backdrop-blur-sm p-5 text-center">
            <div className="text-sm text-fg-muted uppercase tracking-wider mb-1">
              Optimal Total Cost
            </div>
            <div className="text-4xl font-display font-bold text-accent tabular-nums">
              {result
                ? `$${result.total_cost.toLocaleString(undefined, {
                    minimumFractionDigits: 0,
                    maximumFractionDigits: 0,
                  })}`
                : "--"}
            </div>
            {result && (
              <div className="text-xs text-fg-faint mt-2">
                Status: {result.status}
              </div>
            )}
          </div>

          {/* Flow summary */}
          {result && result.flows && (
            <div className="rounded-card border border-surface-border bg-surface/30 backdrop-blur-sm p-5 space-y-3">
              <h3 className="text-sm font-medium text-fg-muted uppercase tracking-wider">
                Flow Summary
              </h3>
              <div className="space-y-2">
                {result.flows
                  .filter((f) => f.flow > 0)
                  .sort((a, b) => b.flow - a.flow)
                  .map((f) => {
                    const fromNode = DEFAULT_NODES.find(
                      (n) => n.id === f.from
                    );
                    const toNode = DEFAULT_NODES.find((n) => n.id === f.to);
                    const utilPct = (f.utilization * 100).toFixed(0);

                    return (
                      <div
                        key={`${f.from}-${f.to}`}
                        className="flex items-center justify-between text-xs"
                      >
                        <span className="text-fg-muted">
                          {fromNode?.label ?? f.from} {"->"}{" "}
                          {toNode?.label ?? f.to}
                        </span>
                        <span className="font-mono text-fg tabular-nums">
                          {Math.round(f.flow)}{" "}
                          <span className="text-fg-faint">({utilPct}%)</span>
                        </span>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Solver status */}
      <SolverStatus
        status={status}
        result={result}
        error={error}
        onRetry={retry}
      />

      {/* Analogs note */}
      <AnalogsBanner />
    </div>
  );
}
