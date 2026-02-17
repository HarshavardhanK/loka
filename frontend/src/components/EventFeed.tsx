import type { MissionEvent } from "../api/contracts";

function severityClass(severity: MissionEvent["severity"]): string {
  if (severity === "critical") {
    return "critical";
  }
  if (severity === "warning") {
    return "warning";
  }
  return "info";
}

export default function EventFeed({ events }: { events: MissionEvent[] }): JSX.Element {
  return (
    <section className="panel event-panel reveal-delayed">
      <header className="panel-header small">
        <p className="eyebrow">Event Log</p>
        <h3>Mission Timeline Events</h3>
      </header>

      <div className="event-list">
        {events.length === 0 ? (
          <p className="empty-state">No mission events yet.</p>
        ) : (
          events
            .slice()
            .reverse()
            .map((event) => (
              <article key={event.id} className="event-item">
                <div className={`severity-dot ${severityClass(event.severity)}`} />
                <div>
                  <p className="event-title">
                    D+{event.missionDay} · {event.title}
                  </p>
                  <p className="event-detail">{event.detail}</p>
                </div>
              </article>
            ))
        )}
      </div>
    </section>
  );
}
