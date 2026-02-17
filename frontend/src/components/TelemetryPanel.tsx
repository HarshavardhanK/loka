import type { SimulationSnapshot } from "../api/contracts";

const SPEED_OPTIONS = [1, 4, 12, 28];

function phaseLabel(phase: SimulationSnapshot["phase"]): string {
  switch (phase) {
    case "launch":
      return "Launch";
    case "escape":
      return "Escape";
    case "cruise":
      return "Cruise";
    case "jovian_approach":
      return "Jovian Approach";
    case "jovian_capture":
      return "Jovian Capture";
    case "science":
      return "Science Orbit";
    default:
      return phase;
  }
}

export default function TelemetryPanel({
  snapshot,
  playing,
  tickDays,
  onTogglePlayback,
  onAdvance,
  onTickDaysChange,
}: {
  snapshot: SimulationSnapshot | null;
  playing: boolean;
  tickDays: number;
  onTogglePlayback: () => void;
  onAdvance: (days: number) => void;
  onTickDaysChange: (days: number) => void;
}): JSX.Element {
  if (!snapshot) {
    return <section className="panel telemetry-panel loading">Initializing mission state...</section>;
  }

  const { spacecraft } = snapshot;

  return (
    <section className="panel telemetry-panel reveal">
      <header className="panel-header">
        <p className="eyebrow">Mission</p>
        <h2>{snapshot.missionName}</h2>
        <span className="status-pill">{phaseLabel(snapshot.phase)}</span>
      </header>

      <div className="kpi-grid">
        <article>
          <p>Mission Day</p>
          <strong>{snapshot.missionDay.toFixed(1)}</strong>
        </article>
        <article>
          <p>Speed</p>
          <strong>{spacecraft.heliocentricSpeedKmS.toFixed(2)} km/s</strong>
        </article>
        <article>
          <p>Delta-V Used</p>
          <strong>{spacecraft.deltaVUsedKmS.toFixed(3)} km/s</strong>
        </article>
        <article>
          <p>Propellant</p>
          <strong>{spacecraft.propellantRemainingPct.toFixed(1)}%</strong>
        </article>
        <article>
          <p>Guidance</p>
          <strong>{spacecraft.guidanceMode.toUpperCase()}</strong>
        </article>
        <article>
          <p>Target</p>
          <strong>{spacecraft.target}</strong>
        </article>
      </div>

      <div className="controls-row">
        <button className="btn primary" onClick={onTogglePlayback} type="button">
          {playing ? "Pause" : "Play"}
        </button>
        <button className="btn" onClick={() => onAdvance(6)} type="button">
          +6d
        </button>
        <button className="btn" onClick={() => onAdvance(24)} type="button">
          +24d
        </button>
      </div>

      <div className="controls-row speed-row">
        <p className="inline-label">Autoplay Step</p>
        <div className="speed-options">
          {SPEED_OPTIONS.map((value) => (
            <button
              key={value}
              className={`chip ${tickDays === value ? "active" : ""}`}
              onClick={() => onTickDaysChange(value)}
              type="button"
            >
              {value}d/tick
            </button>
          ))}
        </div>
      </div>

      <p className="timestamp">Simulation UTC: {snapshot.simTimeUtc.replace("T", " ").slice(0, 19)}</p>
    </section>
  );
}
