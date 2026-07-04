import type { NodePosition, EdgeDefinition } from "../types";
import { buildEdgeKey } from "../useNetworkFlow";

interface ControlPanelProps {
  nodes: NodePosition[];
  edges: EdgeDefinition[];
  nodeValues: Record<string, number>;
  edgeCapacities: Record<string, number>;
  onNodeValueChange: (nodeId: string, value: number) => void;
  onEdgeCapacityChange: (fromId: string, toId: string, capacity: number) => void;
  onResetAll: () => void;
}

function SliderRow({
  label,
  value,
  defaultValue,
  max,
  step,
  onChange,
  onReset,
}: {
  label: string;
  value: number;
  defaultValue: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  onReset: () => void;
}) {
  const pct = (value / max) * 100;
  const isDefault = value === defaultValue;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-fg">{label}</label>
        <div className="flex items-center gap-2">
          <span className="text-sm font-mono text-accent tabular-nums w-12 text-right">
            {value}
          </span>
          {!isDefault && (
            <button
              onClick={onReset}
              className="text-xs text-fg-faint hover:text-fg-muted transition-colors px-1.5 py-0.5 rounded border border-surface-border hover:border-fg-faint"
              title="Reset to default"
            >
              Reset
            </button>
          )}
        </div>
      </div>
      <div className="relative group">
        <div className="absolute inset-0 h-2 top-1/2 -translate-y-1/2 rounded-full bg-surface-elevated overflow-hidden pointer-events-none">
          <div
            className="h-full rounded-full bg-gradient-to-r from-accent/40 to-accent/70 transition-all duration-150"
            style={{ width: `${pct}%` }}
          />
        </div>
        <input
          type="range"
          min={0}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="relative w-full h-6 appearance-none bg-transparent cursor-pointer slider-thumb"
        />
      </div>
    </div>
  );
}

export function ControlPanel({
  nodes,
  edges,
  nodeValues,
  edgeCapacities,
  onNodeValueChange,
  onEdgeCapacityChange,
  onResetAll,
}: ControlPanelProps) {
  const supplyNodes = nodes.filter((n) => n.type === "supply");
  const demandNodes = nodes.filter((n) => n.type === "demand");

  const hasChanges =
    nodes.some((n) => (nodeValues[n.id] ?? n.defaultValue) !== n.defaultValue) ||
    edges.some(
      (e) =>
        (edgeCapacities[buildEdgeKey(e.from, e.to)] ?? e.defaultCapacity) !==
        e.defaultCapacity
    );

  return (
    <div className="rounded-card border border-surface-border bg-surface/30 backdrop-blur-sm p-5 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-fg-muted uppercase tracking-wider">
          Controls
        </h2>
        {hasChanges && (
          <button
            onClick={onResetAll}
            className="text-xs font-medium text-accent-secondary hover:text-accent transition-colors px-2 py-1 rounded border border-surface-border hover:border-accent-secondary/50"
          >
            Reset All
          </button>
        )}
      </div>

      {/* Supply section */}
      <div className="space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider" style={{ color: "hsl(160, 84%, 39%)" }}>
          Supply Capacities
        </h3>
        {supplyNodes.map((node) => (
          <SliderRow
            key={node.id}
            label={node.label}
            value={nodeValues[node.id] ?? node.defaultValue}
            defaultValue={node.defaultValue}
            max={node.defaultValue * 2}
            step={10}
            onChange={(v) => onNodeValueChange(node.id, v)}
            onReset={() => onNodeValueChange(node.id, node.defaultValue)}
          />
        ))}
      </div>

      {/* Demand section */}
      <div className="space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider" style={{ color: "hsl(40, 90%, 55%)" }}>
          Demand Requirements
        </h3>
        {demandNodes.map((node) => (
          <SliderRow
            key={node.id}
            label={node.label}
            value={nodeValues[node.id] ?? node.defaultValue}
            defaultValue={node.defaultValue}
            max={node.defaultValue * 2}
            step={10}
            onChange={(v) => onNodeValueChange(node.id, v)}
            onReset={() => onNodeValueChange(node.id, node.defaultValue)}
          />
        ))}
      </div>

      {/* Edge capacities section */}
      <div className="space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider" style={{ color: "hsl(200, 80%, 55%)" }}>
          Route Capacities
        </h3>
        <div className="space-y-3">
          {edges.map((edge) => {
            const key = buildEdgeKey(edge.from, edge.to);
            const fromNode = nodes.find((n) => n.id === edge.from);
            const toNode = nodes.find((n) => n.id === edge.to);
            const label = `${fromNode?.label ?? edge.from} -> ${toNode?.label ?? edge.to}`;

            return (
              <SliderRow
                key={key}
                label={label}
                value={edgeCapacities[key] ?? edge.defaultCapacity}
                defaultValue={edge.defaultCapacity}
                max={edge.defaultCapacity * 2}
                step={10}
                onChange={(v) => onEdgeCapacityChange(edge.from, edge.to, v)}
                onReset={() =>
                  onEdgeCapacityChange(edge.from, edge.to, edge.defaultCapacity)
                }
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
