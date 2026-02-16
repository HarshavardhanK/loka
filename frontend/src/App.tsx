import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createSimulationApi } from "./api/client";
import type { SimulationSnapshot } from "./api/contracts";
import EventFeed from "./components/EventFeed";
import MissionTimeline from "./components/MissionTimeline";
import PromptConsole, { type ChatLine } from "./components/PromptConsole";
import SolarSystemScene from "./components/SolarSystemScene";
import TelemetryPanel from "./components/TelemetryPanel";

const TIMELINE_MAX_DAY = 760;
const AUTOPLAY_INTERVAL_MS = 1_500;

export default function App(): JSX.Element {
  const api = useMemo(() => createSimulationApi(), []);

  const [snapshot, setSnapshot] = useState<SimulationSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [tickDays, setTickDays] = useState(4);
  const [promptBusy, setPromptBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lines, setLines] = useState<ChatLine[]>([]);

  const missionDayRef = useRef(0);
  const busyRef = useRef(false);

  useEffect(() => {
    missionDayRef.current = snapshot?.missionDay ?? 0;
  }, [snapshot?.missionDay]);

  const runSnapshotUpdate = useCallback(async (task: () => Promise<SimulationSnapshot>) => {
    if (busyRef.current) {
      return;
    }

    busyRef.current = true;
    try {
      const nextSnapshot = await task();
      setSnapshot(nextSnapshot);
      setError(null);
    } catch (taskError) {
      const message = taskError instanceof Error ? taskError.message : "Simulation update failed";
      setError(message);
    } finally {
      busyRef.current = false;
      setLoading(false);
    }
  }, []);

  const advance = useCallback(
    async (days: number) => {
      await runSnapshotUpdate(() => api.advance(days));
    },
    [api, runSnapshotUpdate],
  );

  const seek = useCallback(
    async (day: number) => {
      await runSnapshotUpdate(() => api.seek(day));
    },
    [api, runSnapshotUpdate],
  );

  useEffect(() => {
    void runSnapshotUpdate(() => api.bootstrap());
  }, [api, runSnapshotUpdate]);

  useEffect(() => {
    if (!playing || !snapshot) {
      return;
    }

    const interval = window.setInterval(() => {
      void advance(tickDays);
    }, AUTOPLAY_INTERVAL_MS);

    return () => window.clearInterval(interval);
  }, [advance, playing, snapshot, tickDays]);

  const sendPrompt = useCallback(
    async (prompt: string) => {
      const userLine: ChatLine = {
        id: `line-user-${Date.now()}`,
        role: "user",
        content: prompt,
      };
      setLines((existing) => [...existing, userLine]);
      setPromptBusy(true);

      try {
        const result = await api.sendPrompt({ prompt });
        const assistantLine: ChatLine = {
          id: `line-assistant-${Date.now()}`,
          role: "assistant",
          content: result.assistant,
        };
        setLines((existing) => [...existing, assistantLine]);
        await runSnapshotUpdate(() => api.seek(missionDayRef.current));
      } catch (promptError) {
        const message = promptError instanceof Error ? promptError.message : "Prompt execution failed";
        setLines((existing) => [
          ...existing,
          {
            id: `line-error-${Date.now()}`,
            role: "assistant",
            content: `Prompt failed: ${message}`,
          },
        ]);
      } finally {
        setPromptBusy(false);
      }
    },
    [api, runSnapshotUpdate],
  );

  return (
    <div className="app-shell">
      <div className="noise-layer" />
      <header className="topbar">
        <div>
          <p className="brand-kicker">Loka Simulation</p>
          <h1>Jovian Mission Control</h1>
        </div>
        <div className="topbar-right">
          <span className="mode-chip">MODE: {import.meta.env.VITE_SIMULATION_MODE === "live" ? "LIVE" : "MOCK"}</span>
          <span className="mode-chip">ENGINE: SPACEKIT + Typed API</span>
        </div>
      </header>

      <main className="main-layout">
        <section className="scene-column">
          <SolarSystemScene
            snapshot={snapshot}
            daysPerSecond={
              playing
                ? tickDays / (AUTOPLAY_INTERVAL_MS / 1_000)
                : 0
            }
          />
          {loading ? <p className="loading-overlay">Loading mission simulation...</p> : null}
        </section>

        <aside className="control-column">
          <TelemetryPanel
            snapshot={snapshot}
            playing={playing}
            tickDays={tickDays}
            onTogglePlayback={() => setPlaying((state) => !state)}
            onAdvance={(days) => {
              void advance(days);
            }}
            onTickDaysChange={(days) => setTickDays(days)}
          />
          <MissionTimeline
            day={snapshot?.missionDay ?? 0}
            maxDay={TIMELINE_MAX_DAY}
            onSeek={(day) => {
              void seek(day);
            }}
          />
          <EventFeed events={snapshot?.events ?? []} />
          <PromptConsole lines={lines} busy={promptBusy} onSend={sendPrompt} />
        </aside>
      </main>

      {error ? <div className="error-banner">Simulation error: {error}</div> : null}
    </div>
  );
}
