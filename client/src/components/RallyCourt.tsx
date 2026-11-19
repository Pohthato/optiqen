// client/src/components/RallyCourt.tsx
// The rally on the court, as the analysis fitted it (worker/analyse_rally.py): every flight, each
// contact, each landing with its 95 % region, the net, and where both players stood at a chosen
// hit with their 95 % regions. Turned by dragging or with the view buttons; drawn by hand, so
// every line is one the analysis holds. Choosing a finding picks out its shots and moves the
// video to its first evidence frame.
import React, { useMemo, useRef, useState } from "react";

type Half = "near" | "far";
type Floor = { point: [number, number]; covariance: number[][] };

export type RallyShot = {
  shot: number;
  hitter: Half;
  hitMs: number;
  path: Array<[number, number, number, number]>; // time (ms), x, y, z in court metres
  landing: (Floor & { call: string; pIn: number }) | null;
  players: Record<Half, Floor | null> | null;
  name?: { said: string } | null;
};

export type RallyFinding = { id: string; kind: string; shots: number[]; evidence: Array<{ frame: number; timeMs: number }>; text: string };

export type RallyView = { shots: RallyShot[]; findings?: RallyFinding[] };

// The court as worker/geometry/court_model.py has it: x across from the left singles line, y along
// from the near baseline, lines by their centres.
const LENGTH = 13.4;
const SINGLES = 5.18;
const MARGIN = 0.46;
const HALF_LINE = 0.02;
const NET_Y = LENGTH / 2;
const COLUMNS = { dl: -MARGIN + HALF_LINE, sl: HALF_LINE, c: SINGLES / 2, sr: SINGLES - HALF_LINE, dr: SINGLES + MARGIN - HALF_LINE };
const ROWS = [HALF_LINE, 0.76 + HALF_LINE, NET_Y - 1.98 - HALF_LINE, NET_Y + 1.98 + HALF_LINE, LENGTH - 0.76 - HALF_LINE, LENGTH - HALF_LINE];
const POST = 1.55;
const NET_CENTRE = 1.524;
const NET_DEPTH = 0.76;
const CHI2_95_2D = 5.991;
const CENTRE: [number, number, number] = [SINGLES / 2, NET_Y, 0];
const WIDTH = 640;
const HEIGHT = 420;
const DISTANCE = 26; // metres from the court's centre to the eye
const FOCAL = 760;

const COLOURS: Record<Half, string> = { near: "#b7fa59", far: "#7dd3fc" };

export const VIEWS = { above: { yaw: 0, pitch: 89 }, behind: { yaw: 0, pitch: 22 }, side: { yaw: 90, pitch: 18 }, corner: { yaw: 35, pitch: 28 } } as const;

type View = { yaw: number; pitch: number };
type Point3 = [number, number, number];

/** A point in court metres as a pixel, for an eye turned `yaw` degrees about the court's centre
 * (0: behind the near baseline) and raised `pitch` degrees; null when behind the eye. */
export function project([x, y, z]: Point3, { yaw, pitch }: View): [number, number] | null {
  const a = (yaw * Math.PI) / 180;
  const b = (pitch * Math.PI) / 180;
  const dx = x - CENTRE[0];
  const dy = y - CENTRE[1];
  const dz = z - CENTRE[2];
  const across = dx * Math.cos(a) - dy * Math.sin(a);
  const along = dx * Math.sin(a) + dy * Math.cos(a);
  const up = dz * Math.cos(b) + along * Math.sin(b);
  const depth = DISTANCE + along * Math.cos(b) - dz * Math.sin(b);
  if (depth < 1) return null;
  return [WIDTH / 2 + (FOCAL * across) / depth, HEIGHT / 2 - (FOCAL * up) / depth];
}

function line(points: Point3[], view: View): string {
  return points
    .map((point) => project(point, view))
    .filter((pixel): pixel is [number, number] => pixel !== null)
    .map(([u, v], i) => `${i ? "L" : "M"}${u.toFixed(1)} ${v.toFixed(1)}`)
    .join(" ");
}

/** The 95 % region of a point on the floor, as its outline. */
export function region({ point, covariance }: Floor, steps = 36): Point3[] {
  const [[a, b], [, d]] = covariance;
  const l11 = Math.sqrt(Math.max(a, 0));
  const l21 = l11 > 0 ? b / l11 : 0;
  const l22 = Math.sqrt(Math.max(d - l21 * l21, 0));
  const r = Math.sqrt(CHI2_95_2D);
  return Array.from({ length: steps + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / steps;
    const u = r * Math.cos(t);
    const v = r * Math.sin(t);
    return [point[0] + l11 * u, point[1] + l21 * u + l22 * v, 0] as Point3;
  });
}

/** The painted lines, the same for singles and doubles. */
function courtLines(): Point3[][] {
  const rows = ROWS.map((y) => [[COLUMNS.dl, y, 0], [COLUMNS.dr, y, 0]] as Point3[]);
  const sides = Object.entries(COLUMNS)
    .filter(([name]) => name !== "c")
    .map(([, x]) => [[x, ROWS[0], 0], [x, ROWS[5], 0]] as Point3[]);
  const centre = [[[COLUMNS.c, 0, 0], [COLUMNS.c, ROWS[2], 0]], [[COLUMNS.c, ROWS[3], 0], [COLUMNS.c, LENGTH, 0]]] as Point3[][];
  return [...rows, ...sides, ...centre];
}

export function RallyCourt({ rally, onSeek }: { rally: RallyView; onSeek?: (timeMs: number) => void }) {
  const [view, setView] = useState<View>(VIEWS.corner);
  const [chosen, setChosen] = useState<string | null>(null);
  const drag = useRef<{ x: number; y: number; view: View } | null>(null);
  const findings = rally.findings ?? [];
  const finding = findings.find((f) => f.id === chosen) ?? null;
  const picked = new Set(finding?.shots ?? []);
  const focus = finding?.kind === "shot" || finding?.kind === "end" ? rally.shots.find((shot) => shot.shot === finding.shots[0]) ?? null : null;

  const lines = useMemo(() => courtLines().map((points) => line(points, view)), [view]);
  const outline = line([[COLUMNS.dl, 0, 0], [COLUMNS.dr, 0, 0], [COLUMNS.dr, LENGTH, 0], [COLUMNS.dl, LENGTH, 0], [COLUMNS.dl, 0, 0]], view);
  const tape: Point3[] = [[COLUMNS.dl, NET_Y, POST], [COLUMNS.c, NET_Y, NET_CENTRE], [COLUMNS.dr, NET_Y, POST]];
  const net = line([...tape, ...tape.slice().reverse().map(([x, y, z]) => [x, y, z - NET_DEPTH] as Point3), tape[0]], view);

  const choose = (f: RallyFinding) => {
    setChosen(f.id === chosen ? null : f.id);
    if (f.id !== chosen && f.evidence.length && onSeek) onSeek(f.evidence[0].timeMs);
  };

  return (
    <section className="panel-surface rounded-2xl p-5" aria-label="The rally on the court">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <p className="font-mono-data text-[10px] uppercase tracking-[.16em] text-[#b7fa59]">Rally on court</p>
        <div className="flex gap-1">
          {(Object.keys(VIEWS) as Array<keyof typeof VIEWS>).map((name) => (
            <button key={name} type="button" onClick={() => setView(VIEWS[name])} className="rounded-md px-2 py-1 font-mono-data text-[10px] uppercase text-slate-400 hover:bg-white/5 hover:text-white">
              {name}
            </button>
          ))}
        </div>
      </div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full touch-none select-none rounded-xl bg-[#0e171d]"
        role="img"
        aria-label={`${rally.shots.length} ${rally.shots.length === 1 ? "shot" : "shots"} drawn on the court`}
        onPointerDown={(event) => {
          drag.current = { x: event.clientX, y: event.clientY, view };
          (event.target as Element).setPointerCapture?.(event.pointerId);
        }}
        onPointerMove={(event) => {
          if (!drag.current) return;
          const { x, y, view: start } = drag.current;
          setView({ yaw: start.yaw + (event.clientX - x) * 0.4, pitch: Math.max(5, Math.min(89, start.pitch + (event.clientY - y) * 0.3)) });
        }}
        onPointerUp={() => {
          drag.current = null;
        }}
      >
        <path d={outline} fill="#16323a" stroke="none" />
        {lines.map((d, i) => <path key={i} d={d} stroke="#e2e8f0" strokeOpacity={0.55} strokeWidth={1.2} fill="none" />)}
        {rally.shots.map((shot) => {
          const dim = picked.size > 0 && !picked.has(shot.shot);
          const floor = shot.path.map(([, x, y]) => [x, y, 0] as Point3);
          return (
            <g key={shot.shot} opacity={dim ? 0.18 : 1} data-shot={shot.shot}>
              <path d={line(floor, view)} stroke="#000" strokeOpacity={0.35} strokeWidth={1.5} fill="none" />
              <path d={line(shot.path.map(([, x, y, z]) => [x, y, z] as Point3), view)} stroke={COLOURS[shot.hitter]} strokeWidth={picked.has(shot.shot) ? 3 : 1.8} fill="none" />
              {shot.path.length > 0 && (() => {
                const at = project(shot.path[0].slice(1) as Point3, view);
                return at ? <circle cx={at[0]} cy={at[1]} r={3.5} fill={COLOURS[shot.hitter]}><title>{`Shot ${shot.shot}${shot.name ? `: ${shot.name.said}` : ""}`}</title></circle> : null;
              })()}
              {shot.landing && <path d={line(region(shot.landing), view)} fill={shot.landing.call === "out" ? "#f87171" : shot.landing.call === "in" ? "#4ade80" : "#facc15"} fillOpacity={0.35} stroke="none"><title>{`Landing: ${shot.landing.call}`}</title></path>}
            </g>
          );
        })}
        <path d={net} fill="#e2e8f0" fillOpacity={0.12} stroke="#e2e8f0" strokeWidth={1.4} />
        {focus?.players && (Object.entries(focus.players) as Array<[Half, Floor | null]>).map(([half, where]) =>
          where ? <path key={half} d={line(region(where), view)} fill={COLOURS[half]} fillOpacity={0.3} stroke={COLOURS[half]} strokeWidth={1}><title>{`Where the ${half} player stood at shot ${focus.shot}`}</title></path> : null,
        )}
      </svg>
      {findings.length > 0 && (
        <ol className="mt-4 space-y-1">
          {findings.map((f) => (
            <li key={f.id}>
              <button type="button" onClick={() => choose(f)} aria-pressed={f.id === chosen} className={`w-full rounded-lg px-3 py-2 text-left text-[13px] leading-snug transition-colors ${f.id === chosen ? "bg-white/10 text-white" : "text-slate-300 hover:bg-white/5"}`}>
                <span className="mr-2 font-mono-data text-[10px] text-slate-500">{f.id}</span>
                {f.text}
              </button>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
