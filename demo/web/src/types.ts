// --- API Request Types ---

export interface ProductInput {
  name: string;
  profit: number;
}

export interface ResourceInput {
  name: string;
  capacity: number;
  usage: number[];
}

export interface SolveRequest {
  products: ProductInput[];
  resources: ResourceInput[];
}

// --- API Response Types ---

export interface ProductResult {
  name: string;
  quantity: number;
  profit_contribution: number;
}

export interface ResourceResult {
  name: string;
  used: number;
  capacity: number;
  utilization: number;
  shadow_price: number;
  binding: boolean;
}

export interface SolveResponse {
  status: string;
  total_profit: number;
  products: ProductResult[];
  resources: ResourceResult[];
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
