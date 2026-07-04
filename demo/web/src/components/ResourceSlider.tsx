import type { ResourceInput } from "../types";

interface ResourceSliderProps {
  resource: ResourceInput;
  index: number;
  defaultCapacity: number;
  onChange: (index: number, capacity: number) => void;
}

export function ResourceSlider({
  resource,
  index,
  defaultCapacity,
  onChange,
}: ResourceSliderProps) {
  const max = defaultCapacity * 2;
  const pct = (resource.capacity / max) * 100;
  const isDefault = resource.capacity === defaultCapacity;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-fg">{resource.name}</label>
        <div className="flex items-center gap-2">
          <span className="text-sm font-mono text-accent tabular-nums">
            {resource.capacity.toLocaleString()}
          </span>
          {!isDefault && (
            <button
              onClick={() => onChange(index, defaultCapacity)}
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
          step={Math.max(1, Math.round(defaultCapacity / 100))}
          value={resource.capacity}
          onChange={(e) => onChange(index, Number(e.target.value))}
          className="relative w-full h-6 appearance-none bg-transparent cursor-pointer slider-thumb"
        />
      </div>
      <div className="flex justify-between text-xs text-fg-faint">
        <span>0</span>
        <span>{max.toLocaleString()}</span>
      </div>
    </div>
  );
}
