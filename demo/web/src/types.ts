// --- Network Flow API Request Types ---

export interface NodeInput {
  id: string;
  type: "supply" | "hub" | "demand";
  value: number;
}

export interface EdgeInput {
  from: string;
  to: string;
  cost: number;
  capacity: number;
}

export interface NetworkFlowRequest {
  nodes: NodeInput[];
  edges: EdgeInput[];
}

// --- Network Flow API Response Types ---

export interface FlowResult {
  from: string;
  to: string;
  flow: number;
  capacity: number;
  utilization: number;
  cost: number;
  flow_cost: number;
}

export interface NodeBalance {
  id: string;
  type: "supply" | "hub" | "demand";
  value: number;
  net_flow: number;
}

export interface NetworkFlowResponse {
  status: string;
  total_cost: number;
  flows: FlowResult[];
  node_balances: NodeBalance[];
  iterations: number;
  solve_time_ms: number;
  solver: string;
}

export interface SolverInfo {
  solver: string;
  version: string;
  method: string;
}

// --- Internal State ---

export type SolveStatus = "idle" | "loading" | "cold-start" | "solved" | "error";

// --- Graph Layout Types ---

export interface NodePosition {
  id: string;
  label: string;
  type: "supply" | "hub" | "demand";
  x: number;
  y: number;
  defaultValue: number;
}

export interface EdgeDefinition {
  from: string;
  to: string;
  defaultCost: number;
  defaultCapacity: number;
}
