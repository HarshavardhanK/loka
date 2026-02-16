import type { PromptCommand, PromptResult, SimulationApi, SimulationSnapshot } from "./contracts";
import { createMockSimulationApi } from "./mockSimulationApi";

class HttpSimulationApi implements SimulationApi {
  constructor(private readonly baseUrl: string) {}

  async bootstrap(): Promise<SimulationSnapshot> {
    return this.request<SimulationSnapshot>("/simulation/bootstrap", { method: "POST" });
  }

  async advance(deltaMissionDays: number): Promise<SimulationSnapshot> {
    return this.request<SimulationSnapshot>("/simulation/advance", {
      method: "POST",
      body: JSON.stringify({ deltaMissionDays }),
    });
  }

  async seek(missionDay: number): Promise<SimulationSnapshot> {
    return this.request<SimulationSnapshot>("/simulation/seek", {
      method: "POST",
      body: JSON.stringify({ missionDay }),
    });
  }

  async sendPrompt(command: PromptCommand): Promise<PromptResult> {
    return this.request<PromptResult>("/simulation/prompt", {
      method: "POST",
      body: JSON.stringify(command),
    });
  }

  private async request<T>(path: string, init: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init.headers ?? {}),
      },
    });

    if (!response.ok) {
      throw new Error(`API request failed (${response.status}): ${path}`);
    }

    return (await response.json()) as T;
  }
}

export function createSimulationApi(): SimulationApi {
  const mockEnabled = import.meta.env.VITE_SIMULATION_MODE !== "live";
  if (mockEnabled) {
    return createMockSimulationApi();
  }

  const baseUrl = import.meta.env.VITE_SIMULATION_API_BASE_URL ?? "http://localhost:8000";
  return new HttpSimulationApi(baseUrl);
}
