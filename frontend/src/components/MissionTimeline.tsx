export default function MissionTimeline({
  day,
  maxDay,
  onSeek,
}: {
  day: number;
  maxDay: number;
  onSeek: (day: number) => void;
}): JSX.Element {
  return (
    <section className="panel timeline-panel reveal-delayed">
      <header className="panel-header small">
        <p className="eyebrow">Temporal Control</p>
        <h3>Mission Timeline</h3>
      </header>

      <div className="timeline-row">
        <span>D+0</span>
        <input
          type="range"
          min={0}
          max={maxDay}
          value={day}
          onChange={(event) => onSeek(Number(event.target.value))}
        />
        <span>D+{maxDay}</span>
      </div>
      <p className="timeline-value">Current: D+{day.toFixed(1)}</p>
    </section>
  );
}
