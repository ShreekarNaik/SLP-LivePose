import { useEffect, useRef, useState } from "react";

type Props = {
  open: boolean;
  deviceIndices: number[];
  onClose: () => void;
  onDone: () => void; // called after successful compute, lets caller proceed
};

type Progress = {
  per_cam_samples: Record<string, number>;
  per_pair_shared: Record<string, number>;
  last_corners_per_cam: Record<string, number>;
  image_sizes: Record<string, [number, number]>;
  ready: boolean;
  cam_ids: number[];
};

const MIN_PER_CAM = 15;
const MIN_PAIR = 8;

export default function CalibrationPanel({ open, deviceIndices, onClose, onDone }: Props) {
  const [phase, setPhase] = useState<"idle" | "capturing" | "computing" | "done" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [params, setParams] = useState({
    squares_x: 5,
    squares_y: 3,
    square_length_mm: 80,
    marker_length_mm: 60,
  });
  const previewBust = useRef(0);

  useEffect(() => {
    if (!open) {
      // best-effort cleanup
      void fetch("/api/calibration/cancel", { method: "POST" });
      setPhase("idle");
      setProgress(null);
      setError(null);
    }
  }, [open]);

  useEffect(() => {
    if (phase !== "capturing") return;
    const id = setInterval(async () => {
      try {
        const r = await fetch("/api/calibration/progress");
        if (!r.ok) return;
        const p: Progress = await r.json();
        setProgress(p);
        previewBust.current++;
      } catch {
        // server may briefly stutter, ignore
      }
    }, 500);
    return () => clearInterval(id);
  }, [phase]);

  async function startCapture() {
    setError(null);
    const r = await fetch("/api/calibration/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        device_indices: deviceIndices,
        ...params,
      }),
    });
    if (!r.ok) {
      setError(await r.text());
      setPhase("error");
      return;
    }
    setPhase("capturing");
  }

  async function compute() {
    setPhase("computing");
    const r = await fetch("/api/calibration/compute", { method: "POST" });
    if (!r.ok) {
      setError(await r.text());
      setPhase("error");
      return;
    }
    setPhase("done");
    onDone();
  }

  async function cancel() {
    await fetch("/api/calibration/cancel", { method: "POST" });
    onClose();
  }

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 bg-bg-0/80 backdrop-blur-sm flex items-center justify-center p-8">
      <div className="panel-raised rounded-lg w-full max-w-5xl max-h-[90vh] flex flex-col">
        <div className="flex items-center justify-between px-5 py-3 border-b border-edge">
          <div className="font-mono text-sm tracking-widest text-teal-400">// CHARUCO CALIBRATION</div>
          <button onClick={cancel} className="text-ink-2 hover:text-ink-0">✕</button>
        </div>

        <div className="flex-1 overflow-auto p-5">
          {phase === "idle" && (
            <SetupView
              params={params}
              setParams={setParams}
              deviceIndices={deviceIndices}
              onStart={startCapture}
            />
          )}
          {(phase === "capturing" || phase === "computing") && progress && (
            <CapturingView
              progress={progress}
              phase={phase}
              previewBust={previewBust.current}
            />
          )}
          {phase === "error" && (
            <div className="text-red-400 font-mono text-sm whitespace-pre-wrap">{error}</div>
          )}
          {phase === "done" && (
            <div className="text-teal-400 font-mono text-sm">
              Calibration saved. Closing…
            </div>
          )}
        </div>

        <div className="px-5 py-3 border-t border-edge flex justify-between gap-2">
          <button className="btn" onClick={cancel}>Cancel</button>
          {phase === "idle" && (
            <button className="btn-primary" onClick={startCapture}>Start Capture</button>
          )}
          {phase === "capturing" && (
            <button
              className="btn-primary disabled:opacity-40"
              disabled={!progress?.ready}
              onClick={compute}
            >
              {progress?.ready ? "Compute Calibration" : "Keep moving the board…"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function SetupView({
  params,
  setParams,
  deviceIndices,
  onStart,
}: {
  params: { squares_x: number; squares_y: number; square_length_mm: number; marker_length_mm: number };
  setParams: (p: typeof params) => void;
  deviceIndices: number[];
  onStart: () => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-6">
      <div>
        <div className="label mb-2">Board Configuration</div>
        <div className="space-y-2">
          {(["squares_x", "squares_y", "square_length_mm", "marker_length_mm"] as const).map((k) => (
            <label key={k} className="flex items-center gap-2 text-xs font-mono">
              <span className="w-44 text-ink-2 uppercase tracking-widest">{k.replace("_", " ")}</span>
              <input
                type="number"
                value={params[k]}
                onChange={(e) => setParams({ ...params, [k]: parseFloat(e.target.value) })}
                className="w-24 bg-bg-0 border border-edge rounded px-2 py-1 focus:outline-none focus:border-teal-600"
              />
            </label>
          ))}
        </div>
        <div className="mt-4 text-xs text-ink-2 font-mono leading-relaxed">
          <p>Print a charuco board matching these parameters.</p>
          <p>Mount it on a rigid flat surface (cardboard works).</p>
          <p>You can fetch a printable PNG via{" "}
            <a className="text-teal-400 underline" href="/api/calibration/board.png" target="_blank">/api/calibration/board.png</a>.
          </p>
        </div>
      </div>

      <div>
        <div className="label mb-2">Devices to Calibrate</div>
        <div className="font-mono text-sm space-y-1">
          {deviceIndices.map((i) => (
            <div key={i} className="px-2 py-1 border border-edge rounded">/dev/video{i}</div>
          ))}
        </div>
        <div className="mt-4 text-xs text-ink-2 font-mono leading-relaxed">
          <p className="mb-1">Capture protocol:</p>
          <ol className="list-decimal list-inside space-y-1">
            <li>Hold the board so all enabled cameras can see it.</li>
            <li>Slowly move + tilt the board through different angles & positions.</li>
            <li>Aim for ≥{MIN_PER_CAM} samples per camera and ≥{MIN_PAIR} shared frames per pair.</li>
            <li>Click "Compute Calibration" when ready.</li>
          </ol>
        </div>
        <button className="btn-primary mt-6" onClick={onStart}>Start Capture →</button>
      </div>
    </div>
  );
}

function CapturingView({
  progress,
  phase,
  previewBust,
}: {
  progress: Progress;
  phase: "capturing" | "computing";
  previewBust: number;
}) {
  return (
    <div>
      {phase === "computing" && (
        <div className="mb-4 text-teal-400 font-mono text-sm animate-pulse">
          Computing calibration… (intrinsics + extrinsics)
        </div>
      )}
      <div className="grid grid-cols-2 gap-4 mb-4">
        {progress.cam_ids.map((cid) => (
          <div key={cid} className="border border-edge rounded overflow-hidden bg-bg-0">
            <img
              src={`/api/calibration/preview/${cid}?t=${previewBust}`}
              alt={`cam ${cid}`}
              className="w-full aspect-video object-cover"
            />
            <div className="p-2 font-mono text-xs">
              <div className="flex items-center justify-between">
                <span className="text-teal-400">CAM {cid}</span>
                <span className="text-ink-2">
                  corners: <span className="text-ink-0">{progress.last_corners_per_cam[cid] ?? 0}</span>
                </span>
              </div>
              <ProgressBar
                value={progress.per_cam_samples[cid] ?? 0}
                target={MIN_PER_CAM}
                label="intrinsic samples"
              />
            </div>
          </div>
        ))}
      </div>

      {Object.keys(progress.per_pair_shared).length > 0 && (
        <div>
          <div className="label mb-2">Shared Views (cam pairs that saw the board together)</div>
          <div className="grid grid-cols-3 gap-2 text-xs font-mono">
            {Object.entries(progress.per_pair_shared).map(([pair, n]) => (
              <div key={pair} className="border border-edge rounded p-2 bg-bg-0">
                <div className="text-ink-2 mb-1">{pair.replace("_", " ↔ ")}</div>
                <ProgressBar value={n} target={MIN_PAIR} label="" />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ProgressBar({
  value,
  target,
  label,
}: {
  value: number;
  target: number;
  label: string;
}) {
  const pct = Math.min(100, (value / target) * 100);
  const done = value >= target;
  return (
    <div>
      <div className="flex justify-between text-[10px] text-ink-2 mb-0.5">
        {label && <span>{label}</span>}
        <span className={done ? "text-teal-400" : ""}>
          {value} / {target}
        </span>
      </div>
      <div className="h-1.5 bg-bg-3 rounded overflow-hidden">
        <div
          className={`h-full transition-all ${done ? "bg-teal-500 shadow-glow" : "bg-teal-700"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
