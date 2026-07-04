import { useState, useEffect, useRef, useCallback } from "react";
import type {
  ProductInput,
  ResourceInput,
  SolveResponse,
  SolveStatus,
} from "./types";

const DEBOUNCE_MS = 300;
const COLD_START_THRESHOLD_MS = 3000;

interface UseProductionMixOptions {
  apiUrl: string;
  products: ProductInput[];
  initialResources: ResourceInput[];
}

interface UseProductionMixReturn {
  resources: ResourceInput[];
  setResourceCapacity: (index: number, capacity: number) => void;
  result: SolveResponse | null;
  status: SolveStatus;
  error: string | null;
  retry: () => void;
}

export function useProductionMix({
  apiUrl,
  products,
  initialResources,
}: UseProductionMixOptions): UseProductionMixReturn {
  const [resources, setResources] = useState<ResourceInput[]>(initialResources);
  const [result, setResult] = useState<SolveResponse | null>(null);
  const [status, setStatus] = useState<SolveStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const coldStartTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortController = useRef<AbortController | null>(null);

  const solve = useCallback(
    async (currentResources: ResourceInput[]) => {
      // Cancel any in-flight request
      if (abortController.current) {
        abortController.current.abort();
      }
      abortController.current = new AbortController();

      setStatus("loading");
      setError(null);

      // If the request takes longer than the threshold, show cold-start state
      coldStartTimer.current = setTimeout(() => {
        setStatus((prev) => (prev === "loading" ? "cold-start" : prev));
      }, COLD_START_THRESHOLD_MS);

      try {
        const response = await fetch(`${apiUrl}/api/solve/production-mix`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            products,
            resources: currentResources,
          }),
          signal: abortController.current.signal,
        });

        if (!response.ok) {
          const text = await response.text();
          throw new Error(
            `Solver returned ${response.status}: ${text || "Unknown error"}`
          );
        }

        const data: SolveResponse = await response.json();
        setResult(data);
        setStatus("solved");
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") {
          return; // Request was cancelled, don't update state
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
    [apiUrl, products]
  );

  const setResourceCapacity = useCallback(
    (index: number, capacity: number) => {
      setResources((prev) => {
        const next = [...prev];
        next[index] = { ...next[index], capacity };
        return next;
      });
    },
    []
  );

  // Debounced solve on resource changes
  useEffect(() => {
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }
    debounceTimer.current = setTimeout(() => {
      solve(resources);
    }, DEBOUNCE_MS);

    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, [resources, solve]);

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
    solve(resources);
  }, [solve, resources]);

  return {
    resources,
    setResourceCapacity,
    result,
    status,
    error,
    retry,
  };
}
