export type MissionPhase =
  | "launch"
  | "escape"
  | "cruise"
  | "jovian_approach"
  | "jovian_capture"
  | "science";

export type EventSeverity = "info" | "warning" | "critical";

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

export interface BodyState {
  id: string;
  name: string;
  kind: "star" | "planet" | "moon" | "asteroid";
  radiusKm: number;
  color: string;
  parentId: string | null;
  orbitRadiusAu: number;
  orbitalPeriodDays: number;
  positionAu: Vector3;
  velocityKmS: Vector3;
}

export interface MissionEvent {
  id: string;
  missionDay: number;
  title: string;
  detail: string;
  severity: EventSeverity;
}

export interface SpacecraftState {
  name: string;
  positionAu: Vector3;
  velocityKmS: Vector3;
  heliocentricSpeedKmS: number;
  propellantRemainingPct: number;
  deltaVUsedKmS: number;
  target: string;
  guidanceMode: "auto" | "manual";
}

export interface SimulationSnapshot {
  sessionId: string;
  missionName: string;
  missionDay: number;
  simTimeUtc: string;
  phase: MissionPhase;
  spacecraft: SpacecraftState;
  bodies: BodyState[];
  trajectoryAu: Vector3[];
  events: MissionEvent[];
}

export interface PromptCommand {
  prompt: string;
}

export interface PromptResult {
  assistant: string;
  createdEvents: MissionEvent[];
}

export interface SimulationApi {
  bootstrap(): Promise<SimulationSnapshot>;
  advance(deltaMissionDays: number): Promise<SimulationSnapshot>;
  seek(missionDay: number): Promise<SimulationSnapshot>;
  sendPrompt(command: PromptCommand): Promise<PromptResult>;
}
