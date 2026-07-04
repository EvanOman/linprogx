import type { NodePosition, EdgeDefinition } from "./types";

export const DEFAULT_NODES: NodePosition[] = [
  // Supply (left column)
  { id: "seattle", label: "Seattle", type: "supply", x: 100, y: 150, defaultValue: 400 },
  { id: "houston", label: "Houston", type: "supply", x: 100, y: 450, defaultValue: 500 },
  // Hubs (middle column)
  { id: "denver", label: "Denver", type: "hub", x: 500, y: 150, defaultValue: 0 },
  { id: "atlanta", label: "Atlanta", type: "hub", x: 500, y: 450, defaultValue: 0 },
  // Demand (right column)
  { id: "nyc", label: "New York", type: "demand", x: 900, y: 100, defaultValue: 300 },
  { id: "chicago", label: "Chicago", type: "demand", x: 900, y: 300, defaultValue: 250 },
  { id: "miami", label: "Miami", type: "demand", x: 900, y: 500, defaultValue: 200 },
];

export const DEFAULT_EDGES: EdgeDefinition[] = [
  { from: "seattle", to: "denver", defaultCost: 5, defaultCapacity: 300 },
  { from: "seattle", to: "chicago", defaultCost: 8, defaultCapacity: 200 },
  { from: "houston", to: "denver", defaultCost: 4, defaultCapacity: 350 },
  { from: "houston", to: "atlanta", defaultCost: 3, defaultCapacity: 400 },
  { from: "denver", to: "nyc", defaultCost: 7, defaultCapacity: 250 },
  { from: "denver", to: "chicago", defaultCost: 3, defaultCapacity: 300 },
  { from: "atlanta", to: "nyc", defaultCost: 6, defaultCapacity: 200 },
  { from: "atlanta", to: "chicago", defaultCost: 5, defaultCapacity: 150 },
  { from: "atlanta", to: "miami", defaultCost: 2, defaultCapacity: 300 },
];
