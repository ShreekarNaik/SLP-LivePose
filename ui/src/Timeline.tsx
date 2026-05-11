import { useEffect, useState } from "react";

type Props = {
  frameIndex: number;
  totalFrames: number;
  fps: number;
  onSeek: (frame: number) => void;
  onTogglePause: (paused: boolean) => void;
  onSpeed: (speed: number) => void;
};

export default function Timeline({
  frameIndex,
  totalFrames,
  fps,
  onSeek,
  onTogglePause,
  onSpeed,
}: Props) {
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const [scrubVal, setScrubVal] = useState<number | null>(null);

  // Keep local scrub state in sync when not actively dragging
  useEffect(() => {
    if (scrubVal === null) {
      // no-op; controlled by frameIndex prop
    }
  }, [frameIndex, scrubVal]);

  if (totalFrames <= 0) {
    return (
      <div className="panel p-2 flex items-center justify-center font-mono text-xs text-ink-2">
        LIVE
      </div>
    );
  }

  const display = scrubVal ?? frameIndex;
  const seconds = display / Math.max(1, fps);

  return (
    <div className="panel p-3 flex items-center gap-3">
      <button
        className="btn-primary w-20"
        onClick={() => {
          const next = !paused;
          setPaused(next);
          onTogglePause(next);
        }}
      >
        {paused ? "▶" : "❚❚"}
      </button>

      <div className="flex flex-col items-center font-mono text-xs">
        <span className="value">{seconds.toFixed(2)}s</span>
        <span className="text-ink-2">
          {display} / {totalFrames}
        </span>
      </div>

      <input
        type="range"
        min={0}
        max={Math.max(0, totalFrames - 1)}
        value={display}
        onChange={(e) => setScrubVal(parseInt(e.target.value, 10))}
        onMouseUp={() => {
          if (scrubVal !== null) onSeek(scrubVal);
          setScrubVal(null);
        }}
        onTouchEnd={() => {
          if (scrubVal !== null) onSeek(scrubVal);
          setScrubVal(null);
        }}
        className="flex-1 accent-teal-500"
      />

      <div className="flex items-center gap-1 font-mono text-xs">
        <span className="text-ink-2">SPEED</span>
        {[0.25, 0.5, 1, 2, 4].map((s) => (
          <button
            key={s}
            className={`px-2 py-0.5 border rounded ${
              speed === s ? "border-teal-500 text-teal-400" : "border-edge text-ink-1"
            }`}
            onClick={() => {
              setSpeed(s);
              onSpeed(s);
            }}
          >
            {s}x
          </button>
        ))}
      </div>
    </div>
  );
}
