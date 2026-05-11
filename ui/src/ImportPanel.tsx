import { useEffect, useState } from "react";
import type { ScanResponse, SequencePreview, SessionInfo } from "./types";

type Props = {
  open: boolean;
  onClose: () => void;
  onImported: (session: SessionInfo) => void;
};

type LiveDevices = number[];

type Mode = "dataset" | "live";

export default function ImportPanel({ open, onClose, onImported }: Props) {
  const [mode, setMode] = useState<Mode>("dataset");

  // Dataset state
  const [path, setPath] = useState("data/mpi_inf_3dhp");
  const [scanResults, setScanResults] = useState<ScanResponse[]>([]);
  const [scanError, setScanError] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const [selectedSeq, setSelectedSeq] = useState<SequencePreview | null>(null);
  const [previewSession, setPreviewSession] = useState<SessionInfo | null>(
    null,
  );
  const [previewing, setPreviewing] = useState(false);
  const [includeGT, setIncludeGT] = useState(true);

  // Live state
  const [liveDevices, setLiveDevices] = useState<LiveDevices>([]);
  const [selectedDevices, setSelectedDevices] = useState<Set<number>>(
    new Set(),
  );
  const [livePreviews, setLivePreviews] = useState<Map<number, string>>(
    new Map(),
  );

  const [calibrationStatus, setCalibrationStatus] = useState<{
    exists: boolean;
    cam_ids: number[];
  }>({
    exists: false,
    cam_ids: [],
  });

  useEffect(() => {
    if (!open) return;
    void fetch("/api/calibration/status")
      .then((r) => r.json())
      .then(setCalibrationStatus);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    if (mode === "live") {
      void refreshLiveDevices();
    }
  }, [open, mode]);

  async function scan() {
    setScanning(true);
    setScanError(null);
    setSelectedSeq(null);
    setPreviewSession(null);
    try {
      const res = await fetch("/api/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      if (!res.ok) {
        setScanError(await res.text());
        setScanResults([]);
        return;
      }
      setScanResults(await res.json());
    } finally {
      setScanning(false);
    }
  }

  async function preview(seq: SequencePreview) {
    setSelectedSeq(seq);
    setPreviewing(true);
    try {
      const res = await fetch("/api/import/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seq_path: seq.seq_path,
          include_ground_truth: includeGT,
        }),
      });
      const session: SessionInfo = await res.json();
      setPreviewSession(session);
    } finally {
      setPreviewing(false);
    }
  }

  async function startImport() {
    if (!selectedSeq) return;
    const res = await fetch("/api/import/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        seq_path: selectedSeq.seq_path,
        include_ground_truth: includeGT,
      }),
    });
    const session: SessionInfo = await res.json();
    onImported(session);
    onClose();
  }

  async function refreshLiveDevices() {
    const r = await fetch("/api/live/devices");
    const list: LiveDevices = await r.json();
    setLiveDevices(list);
    setSelectedDevices(new Set(list)); // default: all selected

    // Generate live previews using getUserMedia (browser-side preview before backend opens)
    const newPreviews = new Map<number, string>();
    for (const idx of list) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: undefined }, // browser doesn't map directly to OpenCV indices
        });
        const track = stream.getVideoTracks()[0];
        const imageCapture = new (window as any).ImageCapture(track);
        const blob = await imageCapture.takePhoto();
        newPreviews.set(idx, URL.createObjectURL(blob));
        track.stop();
      } catch {
        /* preview unavailable; skip */
      }
    }
    setLivePreviews(newPreviews);
  }

  async function startLive() {
    const indices = Array.from(selectedDevices).sort((a, b) => a - b);
    const res = await fetch("/api/live/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device_indices: indices }),
    });
    const session: SessionInfo = await res.json();
    onImported(session);
    onClose();
  }

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 bg-bg-0/80 backdrop-blur-sm flex items-center justify-center p-8">
      <div className="panel-raised rounded-lg w-full max-w-5xl max-h-[90vh] flex flex-col">
        <div className="flex items-center justify-between px-5 py-3 border-b border-edge">
          <div className="font-mono text-sm tracking-widest text-teal-400">
            // IMPORT
          </div>
          <button onClick={onClose} className="text-ink-2 hover:text-ink-0">
            ✕
          </button>
        </div>

        <div className="px-5 py-2 border-b border-edge flex gap-2">
          {(["dataset", "live"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-3 py-1 text-xs font-mono uppercase tracking-widest rounded border ${
                mode === m
                  ? "border-teal-500 text-teal-400 bg-teal-700/20"
                  : "border-edge text-ink-2 hover:text-ink-1"
              }`}
            >
              {m === "dataset" ? "Dataset" : "Live Cameras"}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-auto p-5">
          {mode === "dataset" ? (
            <DatasetMode
              path={path}
              setPath={setPath}
              scan={scan}
              scanning={scanning}
              scanResults={scanResults}
              scanError={scanError}
              selectedSeq={selectedSeq}
              previewSession={previewSession}
              previewing={previewing}
              preview={preview}
              includeGT={includeGT}
              setIncludeGT={setIncludeGT}
            />
          ) : (
            <LiveMode
              devices={liveDevices}
              selected={selectedDevices}
              setSelected={setSelectedDevices}
              previews={livePreviews}
              refresh={refreshLiveDevices}
              calibrationStatus={calibrationStatus}
              onCalibrate={() => {
                onClose();
                window.dispatchEvent(
                  new CustomEvent("livepose:calibrate", {
                    detail: Array.from(selectedDevices),
                  }),
                );
              }}
            />
          )}
        </div>

        <div className="px-5 py-3 border-t border-edge flex justify-end gap-2">
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          {mode === "dataset" ? (
            <button
              className="btn-primary"
              disabled={!selectedSeq}
              onClick={startImport}
            >
              Import & Stream
            </button>
          ) : (
            <button
              className="btn-primary"
              disabled={selectedDevices.size === 0}
              onClick={startLive}
            >
              Start Live ({selectedDevices.size} cam)
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function DatasetMode(props: {
  path: string;
  setPath: (s: string) => void;
  scan: () => void;
  scanning: boolean;
  scanResults: ScanResponse[];
  scanError: string | null;
  selectedSeq: SequencePreview | null;
  previewSession: SessionInfo | null;
  previewing: boolean;
  preview: (s: SequencePreview) => void;
  includeGT: boolean;
  setIncludeGT: (v: boolean) => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <div>
        <label className="label block mb-1">Folder Path</label>
        <div className="flex gap-2 mb-3">
          <input
            type="text"
            className="flex-1 bg-bg-0 border border-edge rounded px-2 py-1 font-mono text-sm focus:outline-none focus:border-teal-600"
            value={props.path}
            onChange={(e) => props.setPath(e.target.value)}
          />
          <button
            className="btn-primary"
            onClick={props.scan}
            disabled={props.scanning}
          >
            {props.scanning ? "..." : "Scan"}
          </button>
        </div>
        {props.scanError && (
          <div className="text-xs text-red-400 font-mono mb-3">
            {props.scanError}
          </div>
        )}
        <div className="space-y-3">
          {props.scanResults.map((ds) => (
            <div key={ds.root}>
              <div className="label mb-1">{ds.kind}</div>
              <div className="space-y-1">
                {ds.sequences.map((s) => (
                  <button
                    key={s.seq_path}
                    onClick={() => props.preview(s)}
                    className={`w-full text-left px-3 py-2 border rounded font-mono text-xs transition-colors ${
                      props.selectedSeq?.seq_path === s.seq_path
                        ? "border-teal-500 bg-teal-700/20"
                        : "border-edge hover:border-teal-700"
                    }`}
                  >
                    <div className="text-ink-0">{s.display_name}</div>
                    <div className="text-ink-2">
                      {s.cam_ids.length} cams · {s.total_frames} frames ·{" "}
                      {s.fps.toFixed(0)} fps
                      {s.has_ground_truth && " · GT"}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div>
        <label className="label block mb-2">Preview</label>
        {props.previewing && (
          <div className="text-ink-2 text-xs font-mono">loading preview...</div>
        )}
        {props.previewSession && (
          <>
            <label className="flex items-center gap-2 mb-2 text-xs font-mono cursor-pointer">
              <input
                type="checkbox"
                checked={props.includeGT}
                disabled={!props.previewSession.has_ground_truth}
                onChange={(e) => props.setIncludeGT(e.target.checked)}
                className="accent-teal-500"
              />
              <span
                className={
                  props.previewSession.has_ground_truth
                    ? "text-ink-1"
                    : "text-ink-2"
                }
              >
                Include ground truth overlay
                {!props.previewSession.has_ground_truth && " (not available)"}
              </span>
            </label>
            <div className="grid grid-cols-3 gap-2">
              {props.previewSession.cameras.map((c) => (
                <div key={c.cam_id} className="relative">
                  <img
                    src={`/api/scan/thumbnail/${c.cam_id}`}
                    alt={`cam ${c.cam_id}`}
                    className="w-full aspect-video object-cover rounded border border-edge"
                  />
                  <div className="absolute top-1 left-1 px-1 bg-bg-0/80 text-teal-400 font-mono text-[10px] rounded">
                    CAM {c.cam_id}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function LiveMode(props: {
  devices: number[];
  selected: Set<number>;
  setSelected: (s: Set<number>) => void;
  previews: Map<number, string>;
  refresh: () => void;
  calibrationStatus: { exists: boolean; cam_ids: number[] };
  onCalibrate: () => void;
}) {
  if (props.devices.length === 0) {
    return (
      <div className="text-center text-ink-2 font-mono py-12">
        <div>No cameras detected.</div>
        <button className="btn mt-4" onClick={props.refresh}>
          Re-scan devices
        </button>
      </div>
    );
  }

  const selectedArr = Array.from(props.selected).sort((a, b) => a - b);
  const calCovers =
    props.calibrationStatus.exists &&
    selectedArr.every((c) => props.calibrationStatus.cam_ids.includes(c));

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div className="label">Detected Devices ({props.devices.length})</div>
        <div className="flex items-center gap-3 text-xs font-mono">
          {props.calibrationStatus.exists ? (
            <span className={calCovers ? "text-teal-400" : "text-yellow-400"}>
              {calCovers
                ? `✓ Calibration available (cams ${props.calibrationStatus.cam_ids.join(",")})`
                : `⚠ Existing calibration covers cams ${props.calibrationStatus.cam_ids.join(",")} only`}
            </span>
          ) : (
            <span className="text-yellow-400">
              ⚠ No calibration — triangulation will be approximate
            </span>
          )}
          <button
            className="btn"
            disabled={selectedArr.length === 0}
            onClick={props.onCalibrate}
          >
            Calibrate Selected
          </button>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3">
        {props.devices.map((idx) => {
          const checked = props.selected.has(idx);
          const preview = props.previews.get(idx);
          return (
            <button
              key={idx}
              onClick={() => {
                const next = new Set(props.selected);
                if (next.has(idx)) next.delete(idx);
                else next.add(idx);
                props.setSelected(next);
              }}
              className={`relative aspect-video bg-bg-0 border rounded overflow-hidden ${
                checked ? "border-teal-500 shadow-glow" : "border-edge"
              }`}
            >
              {preview ? (
                <img
                  src={preview}
                  className="w-full h-full object-cover"
                  alt=""
                />
              ) : (
                <div className="flex items-center justify-center h-full text-ink-2 font-mono text-xs">
                  device {idx}
                </div>
              )}
              <div className="absolute top-1 left-1 px-1 bg-bg-0/80 font-mono text-[10px] text-teal-400 rounded">
                /dev/video{idx}
              </div>
              <div className="absolute top-1 right-1">
                <input
                  type="checkbox"
                  checked={checked}
                  readOnly
                  className="accent-teal-500"
                />
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
