import { useEffect, useRef, useState } from "react";

type BackendInfo = {
  name: string;
  variants: string[];
  supports_batch: boolean;
  installed: boolean;
};

type ProcessingConfig = {
  smoothing_enabled: boolean;
  min_cutoff: number;
  beta: number;
  imgsz: number;
  conf_threshold: number;
  backend: string;
  backend_variant: string;
  detection_max_distance_px: number;
  epipolar_threshold_px: number;
  bone_ik_enabled: boolean;
  bone_phase?: string;
};

type Props = {
  sessionActive: boolean;
};

export default function ProcessingPanel({ sessionActive }: Props) {
  const [open, setOpen] = useState(false);
  const [config, setConfig] = useState<ProcessingConfig | null>(null);
  const [backends, setBackends] = useState<BackendInfo[]>([]);
  const [swapping, setSwapping] = useState(false);
  const [resetting, setResetting] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Fetch current config + backend list when panel opens or session becomes active
  useEffect(() => {
    if (!sessionActive || !open) return;
    Promise.all([
      fetch("/api/control/processing").then((r) => r.json()),
      fetch("/api/control/backends").then((r) => r.json()),
    ]).then(([cfg, bks]) => {
      setConfig(cfg as ProcessingConfig);
      setBackends(bks as BackendInfo[]);
    }).catch(console.error);
  }, [sessionActive, open]);

  function send(patch: Partial<ProcessingConfig>, immediate = false) {
    if (!config) return;
    const next = { ...config, ...patch };
    setConfig(next);

    if (debounceRef.current) clearTimeout(debounceRef.current);
    const doSend = () =>
      fetch("/api/control/processing", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      })
        .then((r) => r.json())
        .then((res) => {
          if (res.config) setConfig(res.config as ProcessingConfig);
        })
        .catch(console.error);

    if (immediate) {
      void doSend();
    } else {
      debounceRef.current = setTimeout(() => void doSend(), 300);
    }
  }

  async function resetEstimators() {
    setResetting(true);
    try {
      await fetch("/api/control/reset-filters", { method: "POST" });
      // Refresh config to get updated bone_phase
      const cfg = await fetch("/api/control/processing").then((r) => r.json());
      setConfig(cfg as ProcessingConfig);
    } finally {
      setResetting(false);
    }
  }

  async function swapBackend(name: string, variant: string) {
    setSwapping(true);
    try {
      const res = await fetch("/api/control/processing", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ backend: name, backend_variant: variant }),
      });
      const data = await res.json();
      if (!res.ok) {
        alert(`Backend swap failed: ${data.detail ?? "unknown error"}`);
        return;
      }
      if (data.config) setConfig(data.config as ProcessingConfig);
    } finally {
      setSwapping(false);
    }
  }

  const activeBackend = backends.find((b) => b.name === config?.backend);
  const IMGSZ_OPTIONS = [320, 480, 640];

  return (
    <div
      className={`relative flex transition-all duration-200 ${open ? "w-72" : "w-7"} shrink-0`}
      style={{ minWidth: open ? "18rem" : "1.75rem" }}
    >
      {/* Collapse handle */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="absolute left-0 top-0 bottom-0 w-7 flex items-center justify-center bg-bg-1 border-l border-edge hover:bg-bg-2 z-10 text-teal-400 font-mono text-[10px]"
        title={open ? "Collapse processing" : "Expand processing"}
      >
        <span style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", letterSpacing: "0.1em" }}>
          {open ? "◀ PROCESSING" : "PROCESSING ▶"}
        </span>
      </button>

      {open && (
        <div className="ml-7 flex-1 panel flex flex-col overflow-hidden text-xs font-mono">
          <div className="px-3 py-2 border-b border-edge text-[10px] uppercase tracking-widest text-teal-400 shrink-0">
            Processing Controls
          </div>

          {!sessionActive ? (
            <div className="flex-1 flex items-center justify-center text-ink-2 p-4 text-center">
              Start a session to configure processing
            </div>
          ) : !config ? (
            <div className="flex-1 flex items-center justify-center text-ink-2">Loading…</div>
          ) : (
            <div className="flex-1 overflow-auto p-3 space-y-4">

              {/* ── Backend ── */}
              <Section label="Backend">
                <div className="space-y-2">
                  <div className="flex flex-wrap gap-1.5">
                    {backends.map((b) => (
                      <button
                        key={b.name}
                        disabled={!b.installed || swapping}
                        onClick={() => {
                          const defaultVariant = b.variants[0] ?? "nano";
                          void swapBackend(b.name, defaultVariant);
                        }}
                        title={!b.installed ? `${b.name} not installed` : undefined}
                        className={[
                          "px-2 py-0.5 rounded border text-[10px] uppercase tracking-wider transition-colors",
                          config.backend === b.name
                            ? "border-teal-400 text-teal-400 bg-teal-400/10"
                            : b.installed
                            ? "border-edge text-ink-2 hover:border-teal-400/50 hover:text-ink-1"
                            : "border-edge/30 text-ink-2/40 cursor-not-allowed",
                        ].join(" ")}
                      >
                        {b.name}
                        {!b.installed && " ✗"}
                      </button>
                    ))}
                  </div>

                  {/* Variant selector */}
                  {activeBackend && activeBackend.variants.length > 1 && (
                    <div className="flex items-center gap-2">
                      <span className="text-ink-2 w-14">Variant</span>
                      <select
                        value={config.backend_variant}
                        disabled={swapping}
                        onChange={(e) => void swapBackend(config.backend, e.target.value)}
                        className="flex-1 bg-bg-2 border border-edge rounded px-1.5 py-0.5 text-ink-1 text-xs"
                      >
                        {activeBackend.variants.map((v) => (
                          <option key={v} value={v}>{v}</option>
                        ))}
                      </select>
                    </div>
                  )}

                  {swapping && (
                    <div className="text-yellow-400 text-[10px]">Swapping backend…</div>
                  )}
                </div>
              </Section>

              {/* ── Performance ── */}
              <Section label="Performance">
                <div className="space-y-2">
                  {/* Resolution — only relevant for backends that show it */}
                  <div className="flex items-center gap-2">
                    <span className="text-ink-2 w-14">imgsz</span>
                    <div className="flex gap-1">
                      {IMGSZ_OPTIONS.map((sz) => (
                        <button
                          key={sz}
                          onClick={() => send({ imgsz: sz }, true)}
                          className={[
                            "px-2 py-0.5 rounded border text-[10px]",
                            config.imgsz === sz
                              ? "border-teal-400 text-teal-400 bg-teal-400/10"
                              : "border-edge text-ink-2 hover:border-teal-400/50",
                          ].join(" ")}
                        >
                          {sz}
                        </button>
                      ))}
                    </div>
                  </div>

                  <SliderRow
                    label="Conf"
                    value={config.conf_threshold}
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    format={(v) => v.toFixed(2)}
                    onChange={(v) => send({ conf_threshold: v })}
                  />
                </div>
              </Section>

              {/* ── Smoothing ── */}
              <Section label="Smoothing">
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-ink-2 w-16">Enabled</span>
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={config.smoothing_enabled}
                        onChange={(e) => send({ smoothing_enabled: e.target.checked }, true)}
                        className="accent-teal-500"
                      />
                      <span className={config.smoothing_enabled ? "text-teal-400" : "text-ink-2"}>
                        {config.smoothing_enabled ? "on" : "off"}
                      </span>
                    </label>
                  </div>

                  <SliderRow
                    label="Smoothness"
                    value={config.min_cutoff}
                    min={0.1}
                    max={5.0}
                    step={0.05}
                    format={(v) => v.toFixed(2)}
                    onChange={(v) => send({ min_cutoff: v })}
                    disabled={!config.smoothing_enabled}
                    hint="min_cutoff (Hz)"
                  />

                  <SliderRow
                    label="Response"
                    value={config.beta}
                    min={0.0}
                    max={0.1}
                    step={0.001}
                    format={(v) => v.toFixed(3)}
                    onChange={(v) => send({ beta: v })}
                    disabled={!config.smoothing_enabled}
                    hint="beta"
                  />
                </div>
              </Section>

              {/* ── Tracking ── */}
              <Section label="Tracking">
                <div className="space-y-2">
                  <SliderRow
                    label="Max Δ px"
                    value={config.detection_max_distance_px}
                    min={30}
                    max={400}
                    step={10}
                    format={(v) => `${v.toFixed(0)}px`}
                    onChange={(v) => send({ detection_max_distance_px: v })}
                    hint="Max 2D distance for detection matching (px)"
                  />
                  <SliderRow
                    label="Epipolar"
                    value={config.epipolar_threshold_px}
                    min={2}
                    max={50}
                    step={1}
                    format={(v) => `${v.toFixed(0)}px`}
                    onChange={(v) => send({ epipolar_threshold_px: v })}
                    hint="Epipolar consistency threshold (px)"
                  />
                </div>
              </Section>

              {/* ── Bone IK ── */}
              <Section label="Bone IK">
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-ink-2 w-16">Enabled</span>
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={config.bone_ik_enabled}
                        onChange={(e) => send({ bone_ik_enabled: e.target.checked }, true)}
                        className="accent-teal-500"
                      />
                      <span className={config.bone_ik_enabled ? "text-teal-400" : "text-ink-2"}>
                        {config.bone_ik_enabled ? "on" : "off"}
                      </span>
                    </label>
                  </div>
                  {config.bone_phase && (
                    <div className="flex items-center gap-2">
                      <span className="text-ink-2 w-16">Phase</span>
                      <span className={[
                        "text-[10px] px-1.5 py-0.5 rounded border font-mono",
                        config.bone_phase === "locked"
                          ? "border-teal-400 text-teal-400"
                          : config.bone_phase === "stabilizing"
                          ? "border-yellow-400 text-yellow-400"
                          : "border-edge text-ink-2",
                      ].join(" ")}>
                        {config.bone_phase}
                      </span>
                    </div>
                  )}
                  <button
                    onClick={() => void resetEstimators()}
                    disabled={resetting}
                    className="w-full px-2 py-1 rounded border border-edge text-ink-2 text-[10px] hover:border-teal-400/50 hover:text-ink-1 transition-colors disabled:opacity-40"
                  >
                    {resetting ? "Resetting…" : "Reset Estimators"}
                  </button>
                </div>
              </Section>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---- Sub-components ---------------------------------------------------------

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-[9px] uppercase tracking-widest text-ink-2 mb-1.5 border-b border-edge/30 pb-0.5">
        {label}
      </div>
      {children}
    </div>
  );
}

function SliderRow({
  label, value, min, max, step, format, onChange, disabled, hint,
}: {
  label: string;
  value: number | undefined;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
  disabled?: boolean;
  hint?: string;
}) {
  if (value === undefined) return null;
  return (
    <div className={`space-y-0.5 ${disabled ? "opacity-40" : ""}`}>
      <div className="flex justify-between items-center">
        <span className="text-ink-2 text-[10px]" title={hint}>{label}</span>
        <span className="text-teal-300 text-[10px] tabular-nums">{format(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-teal-500 cursor-pointer"
      />
    </div>
  );
}
