/**
 * Tiny SVG sparkline — no external deps.
 * Renders a polyline over a fixed-length number array.
 */

type Props = {
  data: number[];
  color?: string;
  height?: number;
  width?: number;
  min?: number;
  max?: number;
};

export default function Sparkline({
  data,
  color = "#00ffd5",
  height = 28,
  width = 80,
  min,
  max,
}: Props) {
  if (data.length < 2) {
    return <svg width={width} height={height} />;
  }

  const lo = min ?? Math.min(...data);
  const hi = max ?? Math.max(...data);
  const range = hi - lo || 1;

  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height - ((v - lo) / range) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
        opacity={0.85}
      />
    </svg>
  );
}
