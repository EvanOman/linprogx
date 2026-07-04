import type { SolveStatus, NetworkFlowResponse } from "../types";

interface SolverStatusProps {
  status: SolveStatus;
  result: NetworkFlowResponse | null;
  error: string | null;
  onRetry: () => void;
}

function Spinner() {
  return (
    <svg
      className="animate-spin h-4 w-4 text-accent-secondary"
      viewBox="0 0 24 24"
      fill="none"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="3"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}

function PulsingDot() {
  return (
    <span className="relative flex h-2.5 w-2.5">
      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-secondary opacity-75" />
      <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-accent-secondary" />
    </span>
  );
}

export function SolverStatus({
  status,
  result,
  error,
  onRetry,
}: SolverStatusProps) {
  return (
    <div className="rounded-card border border-surface-border bg-surface/30 backdrop-blur-sm px-4 py-3">
      {status === "idle" && (
        <div className="flex items-center gap-2 text-sm text-fg-faint">
          <Spinner />
          <span>Initializing solver...</span>
        </div>
      )}

      {status === "loading" && (
        <div className="flex items-center gap-2 text-sm text-fg-muted">
          <Spinner />
          <span>Solving...</span>
        </div>
      )}

      {status === "cold-start" && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm text-accent-secondary">
            <PulsingDot />
            <span>Waking up the solver...</span>
          </div>
          <p className="text-xs text-fg-faint pl-5">
            The solver runs on a free tier and spins down after inactivity. The
            first request may take 30-60 seconds.
          </p>
        </div>
      )}

      {status === "error" && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-red-400">
              Failed to solve: {error}
            </span>
            <button
              onClick={onRetry}
              className="text-sm font-medium text-accent-secondary hover:text-accent transition-colors px-3 py-1 rounded border border-surface-border hover:border-accent-secondary/50"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {status === "solved" && result && (
        <div className="flex items-center justify-between flex-wrap gap-2">
          <span className="text-sm text-fg-muted">
            Solved by{" "}
            <a
              href="https://github.com/EvanOman/linprogx"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent-secondary hover:text-accent transition-colors font-mono"
            >
              {result.solver}
            </a>{" "}
            in{" "}
            <span className="font-mono text-fg">
              {result.solve_time_ms.toFixed(2)}ms
            </span>
            {" | "}
            <span className="font-mono text-fg-faint">
              {result.iterations} iteration{result.iterations !== 1 ? "s" : ""}
            </span>
          </span>
        </div>
      )}
    </div>
  );
}
