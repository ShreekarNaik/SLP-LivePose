import { useState } from "react";
import Sparkline from "./Sparkline";
import type { MetricsState } from "./metrics";

type Props = {
  metrics: MetricsState;
  stageTiming?: Record<string, number>;
};

export default function MetricsPanel({ metrics, stageTiming }: Props) {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<"overall" | "bones" | "joints" | "cameras" | "pipeline">("overall");

  return (
    <div
      className={`relative flex transition-all duration-200 ${open ? "w-80" : "w-7"} shrink-0`}
      style={{ minWidth: open ? "20rem" : "1.75rem" }}
    >
      {/* Collapse handle */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="absolute left-0 top-0 bottom-0 w-7 flex items-center justify-center bg-bg-1 border-l border-edge hover:bg-bg-2 z-10 text-teal-400 font-mono text-[10px] writing-vertical"
        title={open ? "Collapse metrics" : "Expand metrics"}
      >
        <span style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", letterSpacing: "0.1em" }}>
          {open ? "◀ METRICS" : "METRICS ▶"}
        </span>
      </button>

      {open && (
        <div className="ml-7 flex-1 panel flex flex-col overflow-hidden text-xs font-mono">
          {/* Tab bar */}
          <div className="flex border-b border-edge shrink-0">
            {(["overall", "bones", "joints", "cameras", "pipeline"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-2 py-1.5 capitalize text-[10px] tracking-wider ${
                  tab === t ? "text-teal-400 border-b border-teal-400" : "text-ink-2 hover:text-ink-1"
                }`}
              >
                {t}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-auto p-2 space-y-2">
            {tab === "overall" && <OverallTab m={metrics.overall} />}
            {tab === "bones" && <BonesTab bones={metrics.bones} />}
            {tab === "joints" && <JointsTab joints={metrics.joints} />}
            {tab === "cameras" && <CamerasTab cameras={metrics.cameras} />}
            {tab === "pipeline" && <PipelineTab timing={stageTiming} />}
          </div>
        </div>
      )}
    </div>
  );
}

// ---- Overall tab ------------------------------------------------------------

function OverallTab({ m }: { m: MetricsState["overall"] }) {
  return (
    <table className="w-full border-collapse">
      <tbody>
        <MetricRow label="FPS" value={m.fps.toFixed(1)} spark={m.fpsHistory} color="#00ffd5" />
        <MetricRow
          label="MPJPE"
          value={m.mpjpe != null ? `${m.mpjpe.toFixed(0)} mm` : "—"}
          spark={m.mpjpeHistory}
          color="#ff8a3d"
        />
        <MetricRow
          label="Det rate"
          value={`${(m.detectionRate * 100).toFixed(0)}%`}
          spark={m.detRateHistory}
          min={0}
          max={1}
          color="#a78bfa"
        />
      </tbody>
    </table>
  );
}

// ---- Bones tab --------------------------------------------------------------

function BonesTab({ bones }: { bones: MetricsState["bones"] }) {
  return (
    <table className="w-full border-collapse">
      <thead>
        <tr className="text-ink-2 text-[9px] uppercase tracking-widest">
          <th className="text-left pb-1">Bone</th>
          <th className="text-right pb-1">Mean</th>
          <th className="text-right pb-1">CV</th>
          <th className="pb-1" />
        </tr>
      </thead>
      <tbody>
        {bones.map((b) => (
          <tr key={b.bone} className="border-t border-edge/30">
            <td className="py-1 text-ink-1 text-[10px] pr-1 truncate max-w-[80px]">{b.bone}</td>
            <td className="py-1 text-right text-teal-300">{b.meanLen > 0 ? `${b.meanLen.toFixed(0)}` : "—"}</td>
            <td className={`py-1 text-right ${b.cv > 0.05 ? "text-orange-400" : "text-teal-300"}`}>
              {b.meanLen > 0 ? `${(b.cv * 100).toFixed(1)}%` : "—"}
            </td>
            <td className="py-1 pl-1">
              <Sparkline data={b.history} height={20} width={60} color="#00ffd5" />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---- Joints tab -------------------------------------------------------------

function JointsTab({ joints }: { joints: MetricsState["joints"] }) {
  return (
    <table className="w-full border-collapse">
      <thead>
        <tr className="text-ink-2 text-[9px] uppercase tracking-widest">
          <th className="text-left pb-1">Joint</th>
          <th className="text-right pb-1">Jitter</th>
          <th className="text-right pb-1">Valid</th>
          <th className="pb-1" />
        </tr>
      </thead>
      <tbody>
        {joints.map((j) => (
          <tr key={j.joint} className="border-t border-edge/30">
            <td className="py-1 text-ink-1 text-[10px] pr-1">{j.joint}</td>
            <td className={`py-1 text-right ${j.jitter > 20 ? "text-orange-400" : "text-teal-300"}`}>
              {j.jitter > 0 ? j.jitter.toFixed(1) : "—"}
            </td>
            <td className="py-1 text-right text-a78bfa" style={{ color: "#a78bfa" }}>
              {`${(j.validPct * 100).toFixed(0)}%`}
            </td>
            <td className="py-1 pl-1">
              <Sparkline data={j.history.slice(-60)} height={20} width={60} color="#a78bfa" />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---- Cameras tab ------------------------------------------------------------

function CamerasTab({ cameras }: { cameras: MetricsState["cameras"] }) {
  if (cameras.length === 0) {
    return <div className="text-ink-2 py-4 text-center">No reprojection data yet</div>;
  }
  return (
    <table className="w-full border-collapse">
      <thead>
        <tr className="text-ink-2 text-[9px] uppercase tracking-widest">
          <th className="text-left pb-1">Cam</th>
          <th className="text-right pb-1">Err px</th>
          <th className="text-right pb-1 max-w-[60px]">Worst</th>
          <th className="pb-1" />
        </tr>
      </thead>
      <tbody>
        {cameras.map((c) => (
          <tr key={c.camId} className="border-t border-edge/30">
            <td className="py-1 text-teal-400">{c.camId}</td>
            <td className={`py-1 text-right ${c.meanErr > 20 ? "text-orange-400" : "text-teal-300"}`}>
              {c.meanErr.toFixed(1)}
            </td>
            <td className="py-1 text-right text-ink-2 text-[9px] truncate max-w-[60px]">{c.worstJoint}</td>
            <td className="py-1 pl-1">
              <Sparkline data={c.history} height={20} width={60} color="#ff00ff" />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---- Pipeline tab -----------------------------------------------------------

function PipelineTab({ timing }: { timing?: Record<string, number> }) {
  if (!timing || Object.keys(timing).length === 0) {
    return <div className="text-ink-2 py-4 text-center">No timing data yet</div>;
  }
  const total = timing["total"] ?? Object.values(timing).reduce((s, v) => s + v, 0);
  return (
    <table className="w-full border-collapse">
      <thead>
        <tr className="text-ink-2 text-[9px] uppercase tracking-widest">
          <th className="text-left pb-1">Stage</th>
          <th className="text-right pb-1">ms</th>
          <th className="text-right pb-1">%</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(timing).map(([stage, ms]) => (
          <tr key={stage} className="border-t border-edge/30">
            <td className="py-1 text-ink-1">{stage}</td>
            <td className="py-1 text-right text-teal-300">{ms.toFixed(1)}</td>
            <td className="py-1 text-right text-ink-2">
              {total > 0 ? `${((ms / total) * 100).toFixed(0)}%` : "—"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---- Reusable row -----------------------------------------------------------

function MetricRow({
  label, value, spark, color = "#00ffd5", min, max,
}: {
  label: string;
  value: string;
  spark: number[];
  color?: string;
  min?: number;
  max?: number;
}) {
  return (
    <tr className="border-t border-edge/30">
      <td className="py-1 text-ink-2 pr-2">{label}</td>
      <td className="py-1 text-teal-300 text-right pr-2 w-20">{value}</td>
      <td className="py-1">
        <Sparkline data={spark} height={20} width={80} color={color} min={min} max={max} />
      </td>
    </tr>
  );
}
