type Props = {
  sessionName: string | null;
  fps: number;
  hasGroundTruth: boolean;
  showGroundTruth: boolean;
  onToggleGT: (v: boolean) => void;
  showRaw2D: boolean;
  onToggleRaw2D: (v: boolean) => void;
  showReproj3D: boolean;
  onToggleReproj3D: (v: boolean) => void;
  showReprojGT: boolean;
  onToggleReprojGT: (v: boolean) => void;
  onOpenImport: () => void;
  onStop: () => void;
  onExport: () => void;
};

export default function Controls({
  sessionName, fps, hasGroundTruth,
  showGroundTruth, onToggleGT,
  showRaw2D, onToggleRaw2D,
  showReproj3D, onToggleReproj3D,
  showReprojGT, onToggleReprojGT,
  onOpenImport, onStop, onExport,
}: Props) {
  return (
    <div className="panel px-4 py-2 flex items-center gap-4 flex-wrap">
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-teal-500 shadow-glow animate-pulse" />
        <span className="font-mono font-bold text-teal-400 tracking-widest">LIVEPOSE</span>
      </div>

      <div className="text-ink-2">|</div>

      <div className="flex items-center gap-3 text-sm">
        {sessionName ? (
          <>
            <span className="label">SESSION</span>
            <span className="value">{sessionName}</span>
            <span className="label">FPS</span>
            <span className="value">{fps.toFixed(1)}</span>
          </>
        ) : (
          <span className="text-ink-2 font-mono text-xs">no session loaded</span>
        )}
      </div>

      <div className="flex-1" />

      {/* Overlay toggles */}
      <div className="flex items-center gap-3 text-xs text-ink-1 font-mono">
        <Toggle label="RAW 2D" value={showRaw2D} onChange={onToggleRaw2D} />
        <Toggle label="REPROJ 3D" value={showReproj3D} onChange={onToggleReproj3D} />
        {hasGroundTruth && (
          <>
            <Toggle label="GT" value={showGroundTruth} onChange={onToggleGT} />
            <Toggle label="REPROJ GT" value={showReprojGT} onChange={onToggleReprojGT} />
          </>
        )}
      </div>

      <div className="text-ink-2">|</div>

      <button className="btn" onClick={onOpenImport}>Import Dataset</button>
      {sessionName && (
        <>
          <button className="btn" onClick={onExport}>Export CSV</button>
          <button className="btn" onClick={onStop}>Stop</button>
        </>
      )}
    </div>
  );
}

function Toggle({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-1.5 cursor-pointer">
      <input
        type="checkbox"
        checked={value}
        onChange={(e) => onChange(e.target.checked)}
        className="accent-teal-500"
      />
      {label}
    </label>
  );
}
