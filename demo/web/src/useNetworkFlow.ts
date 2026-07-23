import { useState, useEffect, useRef, useCallback } from "react";
import type {
  NodeInput,
  EdgeInput,
  NetworkFlowResponse,
  SolveStatus,
  NodePosition,
  EdgeDefinition,
} from "./types";

const DEBOUNCE_MS = 300;
const COLD_START_THRESHOLD_MS = 3000;

interface UseNetworkFlowOptions {
  apiUrl: string;
  defaultNodes: NodePosition[];
  defaultEdges: EdgeDefinition[];
}

interface UseNetworkFlowReturn {
  nodeValues: Record<string, number>;
  edgeCapacities: Record<string, number>;
  setNodeValue: (nodeId: string, value: number) => void;
  setEdgeCapacity: (fromId: string, toId: string, capacity: number) => void;
  resetAll: () => void;
  result: NetworkFlowResponse | null;
  status: SolveStatus;
  error: string | null;
  retry: () => void;
}

function buildEdgeKey(from: string, to: string): string {
  return `${from}:${to}`;
}

export { buildEdgeKey };

export function useNetworkFlow({
  apiUrl,
  defaultNodes,
  defaultEdges,
}: UseNetworkFlowOptions): UseNetworkFlowReturn {
  const [nodeValues, setNodeValues] = useState<Record<string, number>>(() => {
    const values: Record<string, number> = {};
    for (const node of defaultNodes) {
      values[node.id] = node.defaultValue;
    }
    return values;
  });

  const [edgeCapacities, setEdgeCapacities] = useState<Record<string, number>>(
    () => {
      const caps: Record<string, number> = {};
      for (const edge of defaultEdges) {
        caps[buildEdgeKey(edge.from, edge.to)] = edge.defaultCapacity;
      }
      return caps;
    }
  );

  const [result, setResult] = useState<NetworkFlowResponse | null>(null);
  const [status, setStatus] = useState<SolveStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const coldStartTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortController = useRef<AbortController | null>(null);

  const solve = useCallback(
    async (
      currentNodeValues: Record<string, number>,
      currentEdgeCapacities: Record<string, number>
    ) => {
      if (abortController.current) {
        abortController.current.abort();
      }
      abortController.current = new AbortController();

      setStatus("loading");
      setError(null);

      coldStartTimer.current = setTimeout(() => {
        setStatus((prev) => (prev === "loading" ? "cold-start" : prev));
      }, COLD_START_THRESHOLD_MS);

      try {
        const nodes: NodeInput[] = defaultNodes.map((n) => ({
          id: n.id,
          type: n.type,
          value: currentNodeValues[n.id] ?? n.defaultValue,
        }));

        const edges: EdgeInput[] = defaultEdges.map((e) => ({
          from: e.from,
          to: e.to,
          cost: e.defaultCost,
          capacity:
            currentEdgeCapacities[buildEdgeKey(e.from, e.to)] ??
            e.defaultCapacity,
        }));

        const response = await fetch(`${apiUrl}/api/solve/network-flow`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ nodes, edges }),
          signal: abortController.current.signal,
        });

        if (!response.ok) {
          const text = await response.text();
          throw new Error(
            `Solver returned ${response.status}: ${text || "Unknown error"}`
          );
        }

        const data: NetworkFlowResponse = await response.json();
        setResult(data);
        setStatus("solved");
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") {
          return;
        }
        const message =
          err instanceof Error ? err.message : "An unexpected error occurred";
        setError(message);
        setStatus("error");
      } finally {
        if (coldStartTimer.current) {
          clearTimeout(coldStartTimer.current);
          coldStartTimer.current = null;
        }
      }
    },
    [apiUrl, defaultNodes, defaultEdges]
  );

  const setNodeValue = useCallback((nodeId: string, value: number) => {
    setNodeValues((prev) => ({ ...prev, [nodeId]: value }));
  }, []);

  const setEdgeCapacity = useCallback(
    (fromId: string, toId: string, capacity: number) => {
      setEdgeCapacities((prev) => ({
        ...prev,
        [buildEdgeKey(fromId, toId)]: capacity,
      }));
    },
    []
  );

  const resetAll = useCallback(() => {
    const values: Record<string, number> = {};
    for (const node of defaultNodes) {
      values[node.id] = node.defaultValue;
    }
    setNodeValues(values);

    const caps: Record<string, number> = {};
    for (const edge of defaultEdges) {
      caps[buildEdgeKey(edge.from, edge.to)] = edge.defaultCapacity;
    }
    setEdgeCapacities(caps);
  }, [defaultNodes, defaultEdges]);

  // Debounced solve on value changes
  useEffect(() => {
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }
    debounceTimer.current = setTimeout(() => {
      solve(nodeValues, edgeCapacities);
    }, DEBOUNCE_MS);

    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, [nodeValues, edgeCapacities, solve]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortController.current) {
        abortController.current.abort();
      }
      if (coldStartTimer.current) {
        clearTimeout(coldStartTimer.current);
      }
    };
  }, []);

  const retry = useCallback(() => {
    solve(nodeValues, edgeCapacities);
  }, [solve, nodeValues, edgeCapacities]);

  return {
    nodeValues,
    edgeCapacities,
    setNodeValue,
    setEdgeCapacity,
    resetAll,
    result,
    status,
    error,
    retry,
  };
}
