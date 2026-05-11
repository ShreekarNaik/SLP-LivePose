import { useEffect, useState } from "react";
import Controls from "./Controls";
import FeedGrid from "./FeedGrid";
import ImportPanel from "./ImportPanel";
import Scene3D from "./Scene3D";
import Timeline from "./Timeline";
import CalibrationPanel from "./CalibrationPanel";
import MetricsPanel from "./MetricsPanel";
import ProcessingPanel from "./ProcessingPanel";
import VideoModal from "./VideoModal";
import { useWebSocket } from "./socket";
import { useMetrics } from "./metrics";
import type { SessionInfo } from "./types";

export default function App() {
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [importOpen, setImportOpen] = useState(false);
  const [showGT, setShowGT] = useState(false);
  const [showRaw2D, setShowRaw2D] = useState(true);
  const [showReproj3D, setShowReproj3D] = useState(true);
  const [showReprojGT, setShowReprojGT] = useState(false);
  const [selectedCam, setSelectedCam] = useState<number | null>(null);
  const [expandedCam, setExpandedCam] = useState<number | null>(null);

  const [enabledCams, setEnabledCams] = useState<Set<number> | null>(null); // null = all

  const [calibrateOpen, setCalibrateOpen] = useState(false);
  const [calibrateDevices, setCalibrateDevices] = useState<number[]>([]);

  const { frame, prevFrame, receiveTime, meta, status } = useWebSocket(session !== null);
  const metrics = useMetrics(frame, meta);

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent<number[]>).detail;
      setCalibrateDevices(detail);
      setCalibrateOpen(true);
    };
    window.addEventListener("livepose:calibrate", handler);
    return () => window.removeEventListener("livepose:calibrate", handler);
  }, []);

  function toggleCam(camId: number) {
    const allCamIds = frame?.msg.cam_ids ?? [];
    let next: Set<number>;
    if (enabledCams === null) {
      // Currently all enabled — disable this one
      next = new Set(allCamIds.filter((c) => c !== camId));
    } else if (enabledCams.has(camId)) {
      next = new Set(enabledCams);
      next.delete(camId);
    } else {
      next = new Set(enabledCams);
      next.add(camId);
    }
    // Enforce minimum 2 cameras for reliable 3D reconstruction
    if (next.size < 2) return;
    // If all are re-enabled, reset to null
    const finalSet = next.size === allCamIds.length ? null : next;
    setEnabledCams(finalSet);
    void fetch("/api/control/cameras", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: finalSet ? [...finalSet] : null }),
    });
  }

  function stopSession() {
    void fetch("/api/control/stop", { method: "POST" }).then(() => {
      setSession(null);
      setEnabledCams(null);
    });
  }

  async function exportCsv() {
    const res = await fetch("/api/export/csv");
    if (!res.ok) {
      alert("No data to export yet — let the pipeline run first.");
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "livepose_export.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="h-full flex flex-col gap-2 p-2">
      <Controls
        sessionName={session?.name ?? null}
        fps={frame?.msg.fps ?? 0}
        hasGroundTruth={!!session?.has_ground_truth}
        showGroundTruth={showGT}
        onToggleGT={setShowGT}
        showRaw2D={showRaw2D}
        onToggleRaw2D={setShowRaw2D}
        showReproj3D={showReproj3D}
        onToggleReproj3D={setShowReproj3D}
        showReprojGT={showReprojGT}
        onToggleReprojGT={setShowReprojGT}
        onOpenImport={() => setImportOpen(true)}
        onStop={stopSession}
        onExport={exportCsv}
      />

      <div className="flex-1 flex gap-2 min-h-0">
        <div className="w-72 shrink-0">
          <FeedGrid
            frame={frame}
            meta={meta}
            selectedCam={selectedCam}
            onSelect={setSelectedCam}
            showRaw2D={showRaw2D}
            showReproj3D={showReproj3D}
            showReprojGT={showReprojGT}
            showGroundTruth={showGT}
            enabledCams={enabledCams}
            onToggleCam={toggleCam}
            onExpand={setExpandedCam}
          />
        </div>

        <div className="flex-1 panel relative overflow-hidden">
          {session ? (
            <Scene3D
              meta={meta}
              cameras={session.cameras}
              frame={frame}
              prevFrame={prevFrame}
              receiveTime={receiveTime}
              showGroundTruth={showGT}
              enabledCams={enabledCams}
              showRaw2D={showRaw2D}
              showReproj3D={showReproj3D}
              showReprojGT={showReprojGT}
              onExpandCam={setExpandedCam}
            />
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-ink-2 font-mono">
              <div className="mb-3 text-xs uppercase tracking-widest">// no session</div>
              <button className="btn-primary" onClick={() => setImportOpen(true)}>
                Import Dataset / Start Live
              </button>
            </div>
          )}
          {/* Status pill */}
          <div className="absolute top-2 right-2 panel px-2 py-1 font-mono text-[10px]">
            <span className="text-ink-2">WS</span>{" "}
            <span
              className={
                status === "open" ? "text-teal-400" : status === "connecting" ? "text-yellow-400" : "text-red-400"
              }
            >
              {status.toUpperCase()}
            </span>
          </div>
        </div>

        {/* Processing panel — collapsible right side */}
        <ProcessingPanel sessionActive={session !== null} />

        {/* Metrics panel — collapsible right side */}
        <MetricsPanel metrics={metrics} stageTiming={frame?.msg.stage_timings_ms} />
      </div>

      <Timeline
        frameIndex={frame?.msg.frame_index ?? 0}
        totalFrames={frame?.msg.total_frames ?? 0}
        fps={frame?.msg.fps ?? (session?.fps ?? 0)}
        onSeek={(i) => void fetch("/api/control/seek", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frame_index: i }),
        })}
        onTogglePause={(p) => void fetch("/api/control/pause", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ paused: p }),
        })}
        onSpeed={(s) => void fetch("/api/control/speed", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ speed: s }),
        })}
      />

      <ImportPanel
        open={importOpen}
        onClose={() => setImportOpen(false)}
        onImported={(s) => setSession(s)}
      />

      <CalibrationPanel
        open={calibrateOpen}
        deviceIndices={calibrateDevices}
        onClose={() => setCalibrateOpen(false)}
        onDone={() => {
          setCalibrateOpen(false);
          setImportOpen(true);
        }}
      />

      {expandedCam !== null && (
        <VideoModal
          camId={expandedCam}
          frame={frame}
          meta={meta}
          showRaw2D={showRaw2D}
          onToggleRaw2D={setShowRaw2D}
          showReproj3D={showReproj3D}
          onToggleReproj3D={setShowReproj3D}
          showReprojGT={showReprojGT}
          onToggleReprojGT={setShowReprojGT}
          showGroundTruth={showGT}
          onToggleGT={setShowGT}
          onClose={() => setExpandedCam(null)}
        />
      )}
    </div>
  );
}
