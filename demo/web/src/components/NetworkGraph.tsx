import { useState, useMemo } from "react";
import type { NodePosition, EdgeDefinition, FlowResult } from "../types";
import { buildEdgeKey } from "../useNetworkFlow";

interface NetworkGraphProps {
  nodes: NodePosition[];
  edges: EdgeDefinition[];
  nodeValues: Record<string, number>;
  edgeCapacities: Record<string, number>;
  flows: FlowResult[] | null;
}

// --- Color constants ---
const COLOR_SUPPLY = "hsl(160, 84%, 39%)";
const COLOR_SUPPLY_BG = "hsl(160, 84%, 39%, 0.12)";
const COLOR_HUB = "hsl(200, 80%, 55%)";
const COLOR_HUB_BG = "hsl(200, 80%, 55%, 0.12)";
const COLOR_DEMAND = "hsl(40, 90%, 55%)";
const COLOR_DEMAND_BG = "hsl(40, 90%, 55%, 0.12)";
const COLOR_NO_FLOW = "hsl(220, 10%, 25%)";
const COLOR_FLOW_LOW = "hsl(200, 80%, 55%)";
const COLOR_FLOW_NEAR_CAP = "hsl(40, 90%, 55%)";
const COLOR_AT_CAP = "hsl(160, 84%, 45%)";
const COLOR_TEXT = "hsl(60, 10%, 93%)";
const COLOR_TEXT_MUTED = "hsl(220, 10%, 65%)";
const COLOR_TEXT_FAINT = "hsl(220, 10%, 45%)";
const COLOR_SURFACE_ELEVATED = "hsl(220, 18%, 13%)";
const COLOR_SURFACE_BORDER = "hsl(220, 15%, 20%)";
const COLOR_BG = "hsl(220, 20%, 8%)";

// --- Layout constants ---
const SVG_WIDTH = 1000;
const SVG_HEIGHT = 600;
const NODE_RECT_W = 110;
const NODE_RECT_H = 56;
const HUB_RADIUS = 32;

// --- Helpers ---

function getNodeCenter(node: NodePosition): { cx: number; cy: number } {
  return { cx: node.x, cy: node.y };
}

function getNodeColors(type: "supply" | "hub" | "demand") {
  switch (type) {
    case "supply":
      return { stroke: COLOR_SUPPLY, fill: COLOR_SUPPLY_BG, text: COLOR_SUPPLY };
    case "hub":
      return { stroke: COLOR_HUB, fill: COLOR_HUB_BG, text: COLOR_HUB };
    case "demand":
      return { stroke: COLOR_DEMAND, fill: COLOR_DEMAND_BG, text: COLOR_DEMAND };
  }
}

function getEdgeColor(utilization: number): string {
  if (utilization <= 0) return COLOR_NO_FLOW;
  if (utilization >= 1.0) return COLOR_AT_CAP;
  if (utilization >= 0.8) return COLOR_FLOW_NEAR_CAP;
  return COLOR_FLOW_LOW;
}

function getEdgeOpacity(utilization: number): number {
  if (utilization <= 0) return 0.3;
  if (utilization >= 1.0) return 1.0;
  if (utilization >= 0.8) return 0.85;
  if (utilization >= 0.5) return 0.7;
  return 0.4;
}

function getEdgeWidth(flow: number, capacity: number): number {
  if (flow <= 0 || capacity <= 0) return 1.5;
  const ratio = Math.min(flow / capacity, 1);
  return 1.5 + ratio * 6.5; // 1.5 to 8
}

/**
 * Compute a quadratic bezier path between two node centers.
 * Edges going to the same destination or from the same source get offset
 * to avoid overlap.
 */
function computeEdgePath(
  from: NodePosition,
  to: NodePosition,
  edgeIndex: number,
  totalEdgesInGroup: number
): string {
  const { cx: x1, cy: y1 } = getNodeCenter(from);
  const { cx: x2, cy: y2 } = getNodeCenter(to);

  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;

  // Calculate perpendicular offset for parallel edges
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy);
  const nx = -dy / len; // perpendicular normal x
  const ny = dx / len;  // perpendicular normal y

  // Offset from center of the group
  const spread = 35;
  const offset =
    totalEdgesInGroup <= 1
      ? 0
      : (edgeIndex - (totalEdgesInGroup - 1) / 2) * spread;

  // Also add curvature so edges bow outward slightly
  const curvature = 30 + Math.abs(offset) * 0.5;
  const curvatureDir = y2 > y1 ? -1 : y2 < y1 ? 1 : edgeIndex % 2 === 0 ? 1 : -1;

  const cpx = mx + nx * (offset + curvature * curvatureDir * 0.3);
  const cpy = my + ny * (offset + curvature * curvatureDir * 0.3);

  return `M${x1},${y1} Q${cpx},${cpy} ${x2},${y2}`;
}

/**
 * Get a point along a quadratic bezier at parameter t.
 */
function bezierPoint(
  x1: number,
  y1: number,
  cpx: number,
  cpy: number,
  x2: number,
  y2: number,
  t: number
): { x: number; y: number } {
  const mt = 1 - t;
  return {
    x: mt * mt * x1 + 2 * mt * t * cpx + t * t * x2,
    y: mt * mt * y1 + 2 * mt * t * cpy + t * t * y2,
  };
}

function parsePath(d: string): {
  x1: number;
  y1: number;
  cpx: number;
  cpy: number;
  x2: number;
  y2: number;
} {
  // Parse "M100,150 Q300,200 500,150"
  const nums = d.match(/-?[\d.]+/g)!.map(Number);
  return {
    x1: nums[0],
    y1: nums[1],
    cpx: nums[2],
    cpy: nums[3],
    x2: nums[4],
    y2: nums[5],
  };
}

// --- Tooltip ---

interface TooltipState {
  x: number;
  y: number;
  flow: FlowResult;
  visible: boolean;
}

// --- Component ---

export function NetworkGraph({
  nodes,
  edges,
  nodeValues,
  edgeCapacities,
  flows,
}: NetworkGraphProps) {
  const [tooltip, setTooltip] = useState<TooltipState>({
    x: 0,
    y: 0,
    flow: { from: "", to: "", flow: 0, capacity: 0, utilization: 0, cost: 0, flow_cost: 0 },
    visible: false,
  });

  const nodeMap = useMemo(() => {
    const map: Record<string, NodePosition> = {};
    for (const n of nodes) {
      map[n.id] = n;
    }
    return map;
  }, [nodes]);

  const flowMap = useMemo(() => {
    const map: Record<string, FlowResult> = {};
    if (flows) {
      for (const f of flows) {
        map[buildEdgeKey(f.from, f.to)] = f;
      }
    }
    return map;
  }, [flows]);

  // Group edges that share endpoints to compute offsets
  const edgeGroups = useMemo(() => {
    // Group by sorted pair of nodes to detect parallel edges
    const groups: Record<string, number[]> = {};
    edges.forEach((_, i) => {
      const e = edges[i];
      const key = [e.from, e.to].sort().join(":");
      if (!groups[key]) groups[key] = [];
      groups[key].push(i);
    });
    return groups;
  }, [edges]);

  const edgePaths = useMemo(() => {
    const paths: { edge: EdgeDefinition; path: string; index: number }[] = [];
    const indexInGroup: Record<string, number> = {};

    for (const edge of edges) {
      const groupKey = [edge.from, edge.to].sort().join(":");
      const group = edgeGroups[groupKey];
      if (!indexInGroup[groupKey]) indexInGroup[groupKey] = 0;
      const idx = indexInGroup[groupKey]++;
      const fromNode = nodeMap[edge.from];
      const toNode = nodeMap[edge.to];
      if (!fromNode || !toNode) continue;

      const path = computeEdgePath(fromNode, toNode, idx, group.length);
      paths.push({ edge, path, index: idx });
    }
    return paths;
  }, [edges, nodeMap, edgeGroups]);

  function handleEdgeEnter(
    e: React.MouseEvent<SVGPathElement>,
    flow: FlowResult
  ) {
    const svg = e.currentTarget.closest("svg");
    if (!svg) return;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const svgPt = pt.matrixTransform(svg.getScreenCTM()?.inverse());
    setTooltip({ x: svgPt.x, y: svgPt.y - 15, flow, visible: true });
  }

  function handleEdgeLeave() {
    setTooltip((prev) => ({ ...prev, visible: false }));
  }

  return (
    <div className="w-full">
      <svg
        viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
        className="w-full h-auto"
        style={{ maxHeight: "70vh" }}
      >
        <defs>
          {/* Glow filter for active edges */}
          <filter id="edge-glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          {/* Subtle shadow for nodes */}
          <filter id="node-shadow" x="-10%" y="-10%" width="120%" height="120%">
            <feDropShadow dx="0" dy="2" stdDeviation="4" floodColor="black" floodOpacity="0.4" />
          </filter>
        </defs>

        {/* Background grid pattern */}
        <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
          <path
            d="M 50 0 L 0 0 0 50"
            fill="none"
            stroke="hsl(220, 15%, 12%)"
            strokeWidth="0.5"
          />
        </pattern>
        <rect width={SVG_WIDTH} height={SVG_HEIGHT} fill={COLOR_BG} rx="12" />
        <rect width={SVG_WIDTH} height={SVG_HEIGHT} fill="url(#grid)" rx="12" opacity="0.5" />

        {/* Column labels */}
        <text x="100" y="38" textAnchor="middle" fill={COLOR_TEXT_FAINT} fontSize="12" fontWeight="600" letterSpacing="0.1em">
          SUPPLY
        </text>
        <text x="500" y="38" textAnchor="middle" fill={COLOR_TEXT_FAINT} fontSize="12" fontWeight="600" letterSpacing="0.1em">
          HUBS
        </text>
        <text x="900" y="38" textAnchor="middle" fill={COLOR_TEXT_FAINT} fontSize="12" fontWeight="600" letterSpacing="0.1em">
          DEMAND
        </text>

        {/* Edges */}
        {edgePaths.map(({ edge, path }) => {
          const key = buildEdgeKey(edge.from, edge.to);
          const flow = flowMap[key];
          const currentCap =
            edgeCapacities[key] ?? edge.defaultCapacity;
          const flowAmount = flow?.flow ?? 0;
          const utilization = flow?.utilization ?? 0;
          const color = getEdgeColor(utilization);
          const opacity = getEdgeOpacity(utilization);
          const width = getEdgeWidth(flowAmount, currentCap);
          const parsed = parsePath(path);
          const midPt = bezierPoint(
            parsed.x1,
            parsed.y1,
            parsed.cpx,
            parsed.cpy,
            parsed.x2,
            parsed.y2,
            0.5
          );
          const costPt = bezierPoint(
            parsed.x1,
            parsed.y1,
            parsed.cpx,
            parsed.cpy,
            parsed.x2,
            parsed.y2,
            0.15
          );

          const numParticles = flowAmount > 0 ? Math.min(Math.ceil(flowAmount / 80), 4) : 0;
          const particleDuration =
            flowAmount > 0 ? Math.max(1.2, 4 - (flowAmount / currentCap) * 2.5) : 4;

          return (
            <g key={key}>
              {/* Edge background (wider, for hover target) */}
              <path
                d={path}
                fill="none"
                stroke="transparent"
                strokeWidth={Math.max(width + 10, 18)}
                style={{ cursor: "pointer" }}
                onMouseEnter={(e) => {
                  if (flow) handleEdgeEnter(e, flow);
                }}
                onMouseLeave={handleEdgeLeave}
              />
              {/* Edge line */}
              <path
                d={path}
                fill="none"
                stroke={color}
                strokeWidth={width}
                strokeLinecap="round"
                opacity={opacity}
                style={{
                  transition: "stroke 500ms ease, stroke-width 500ms ease, opacity 500ms ease",
                  pointerEvents: "none",
                }}
                filter={utilization >= 0.8 ? "url(#edge-glow)" : undefined}
              />

              {/* Flow particles */}
              {numParticles > 0 &&
                Array.from({ length: numParticles }).map((_, i) => (
                  <circle
                    key={`particle-${key}-${i}`}
                    r={2.5}
                    fill={utilization >= 1.0 ? COLOR_AT_CAP : utilization >= 0.8 ? COLOR_FLOW_NEAR_CAP : COLOR_FLOW_LOW}
                    opacity={0.9}
                  >
                    <animateMotion
                      dur={`${particleDuration}s`}
                      repeatCount="indefinite"
                      path={path}
                      begin={`${(i * particleDuration) / numParticles}s`}
                    />
                  </circle>
                ))}

              {/* Flow/capacity label at midpoint */}
              <text
                x={midPt.x}
                y={midPt.y - 8}
                textAnchor="middle"
                fontSize="11"
                fontFamily="'JetBrains Mono', monospace"
                fill={flowAmount > 0 ? COLOR_TEXT_MUTED : COLOR_TEXT_FAINT}
                style={{ transition: "fill 500ms ease" }}
              >
                {Math.round(flowAmount)}/{currentCap}
              </text>

              {/* Cost label near source */}
              <text
                x={costPt.x}
                y={costPt.y - 8}
                textAnchor="middle"
                fontSize="9"
                fontFamily="'JetBrains Mono', monospace"
                fill={COLOR_TEXT_FAINT}
              >
                ${edge.defaultCost}
              </text>
            </g>
          );
        })}

        {/* Nodes */}
        {nodes.map((node) => {
          const { cx, cy } = getNodeCenter(node);
          const colors = getNodeColors(node.type);
          const value = nodeValues[node.id] ?? node.defaultValue;

          if (node.type === "hub") {
            return (
              <g key={node.id} filter="url(#node-shadow)">
                <circle
                  cx={cx}
                  cy={cy}
                  r={HUB_RADIUS}
                  fill={colors.fill}
                  stroke={colors.stroke}
                  strokeWidth="2"
                />
                <text
                  x={cx}
                  y={cy - 4}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize="13"
                  fontWeight="600"
                  fill={COLOR_TEXT}
                >
                  {node.label}
                </text>
                <text
                  x={cx}
                  y={cy + 14}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize="10"
                  fontFamily="'JetBrains Mono', monospace"
                  fill={colors.text}
                >
                  HUB
                </text>
              </g>
            );
          }

          const halfW = NODE_RECT_W / 2;
          const halfH = NODE_RECT_H / 2;
          const typeLabel = node.type === "supply" ? "Supply" : "Demand";

          return (
            <g key={node.id} filter="url(#node-shadow)">
              <rect
                x={cx - halfW}
                y={cy - halfH}
                width={NODE_RECT_W}
                height={NODE_RECT_H}
                rx="10"
                ry="10"
                fill={colors.fill}
                stroke={colors.stroke}
                strokeWidth="2"
              />
              <text
                x={cx}
                y={cy - 10}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize="13"
                fontWeight="600"
                fill={COLOR_TEXT}
              >
                {node.label}
              </text>
              <text
                x={cx}
                y={cy + 10}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize="12"
                fontFamily="'JetBrains Mono', monospace"
                fill={colors.text}
              >
                {typeLabel}: {value}
              </text>
            </g>
          );
        })}

        {/* Tooltip */}
        {tooltip.visible && (
          <g
            transform={`translate(${Math.min(tooltip.x, SVG_WIDTH - 170)}, ${Math.max(tooltip.y - 70, 10)})`}
            style={{ pointerEvents: "none" }}
          >
            <rect
              x="-80"
              y="-4"
              width="160"
              height="64"
              rx="8"
              fill={COLOR_SURFACE_ELEVATED}
              stroke={COLOR_SURFACE_BORDER}
              strokeWidth="1"
              opacity="0.95"
            />
            <text
              x="0"
              y="14"
              textAnchor="middle"
              fontSize="11"
              fontWeight="600"
              fill={COLOR_TEXT}
            >
              Flow: {Math.round(tooltip.flow.flow)} / {tooltip.flow.capacity}
            </text>
            <text
              x="0"
              y="30"
              textAnchor="middle"
              fontSize="10"
              fontFamily="'JetBrains Mono', monospace"
              fill={COLOR_TEXT_MUTED}
            >
              Cost: ${tooltip.flow.cost}/unit
            </text>
            <text
              x="0"
              y="46"
              textAnchor="middle"
              fontSize="10"
              fontFamily="'JetBrains Mono', monospace"
              fill={COLOR_SUPPLY}
            >
              Total: ${Math.round(tooltip.flow.flow_cost).toLocaleString()}
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}
