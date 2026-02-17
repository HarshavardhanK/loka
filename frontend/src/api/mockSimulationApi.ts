import type {
  MissionEvent,
  PromptCommand,
  PromptResult,
  SimulationApi,
  SimulationSnapshot,
  Vector3,
} from "./contracts";
import { buildBodyStates, formatMissionPhase, getBodyById } from "../catalog/solarCatalog";

const ARRIVAL_DAY = 620;
const MAX_DAY = 760;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function cubicBezier(a: Vector3, b: Vector3, c: Vector3, d: Vector3, t: number): Vector3 {
  const mt = 1 - t;
  const mt2 = mt * mt;
  const t2 = t * t;

  return {
    x: a.x * mt2 * mt + 3 * b.x * mt2 * t + 3 * c.x * mt * t2 + d.x * t2 * t,
    y: a.y * mt2 * mt + 3 * b.y * mt2 * t + 3 * c.y * mt * t2 + d.y * t2 * t,
    z: a.z * mt2 * mt + 3 * b.z * mt2 * t + 3 * c.z * mt * t2 + d.z * t2 * t,
  };
}

function vectorMagnitude(v: Vector3): number {
  return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

function subtract(a: Vector3, b: Vector3): Vector3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function generateTrajectoryPoints(): Vector3[] {
  const launchBodies = buildBodyStates(0);
  const arrivalBodies = buildBodyStates(ARRIVAL_DAY);
  const earth = getBodyById(launchBodies, "earth");
  const jupiter = getBodyById(arrivalBodies, "jupiter");

  if (!earth || !jupiter) {
    return [{ x: 0, y: 0, z: 0 }];
  }

  const start = earth.positionAu;
  const finish = jupiter.positionAu;

  const midRise = 0.34;
  const controlA: Vector3 = {
    x: start.x + 0.7,
    y: midRise,
    z: start.z - 0.5,
  };
  const controlB: Vector3 = {
    x: finish.x - 1.4,
    y: -midRise * 0.7,
    z: finish.z + 0.6,
  };

  const path: Vector3[] = [];
  for (let i = 0; i <= ARRIVAL_DAY; i += 4) {
    const t = i / ARRIVAL_DAY;
    const bend = Math.sin(t * Math.PI) * 0.08;
    const point = cubicBezier(start, controlA, controlB, finish, t);
    path.push({
      x: point.x,
      y: point.y + bend,
      z: point.z,
    });
  }

  for (let i = 0; i < 40; i += 1) {
    const t = (i + 1) / 40;
    path.push({
      x: finish.x + Math.cos(t * Math.PI * 6) * 0.09,
      y: Math.sin(t * Math.PI * 3) * 0.02,
      z: finish.z + Math.sin(t * Math.PI * 6) * 0.09,
    });
  }

  return path;
}

function samplePath(path: Vector3[], missionDay: number): Vector3 {
  if (path.length === 0) {
    return { x: 0, y: 0, z: 0 };
  }

  const normalized = clamp(missionDay / MAX_DAY, 0, 1);
  const index = normalized * (path.length - 1);
  const left = Math.floor(index);
  const right = Math.min(path.length - 1, left + 1);
  const t = index - left;

  return {
    x: lerp(path[left].x, path[right].x, t),
    y: lerp(path[left].y, path[right].y, t),
    z: lerp(path[left].z, path[right].z, t),
  };
}

function estimateVelocity(path: Vector3[], missionDay: number): Vector3 {
  const prev = samplePath(path, Math.max(0, missionDay - 1));
  const next = samplePath(path, Math.min(MAX_DAY, missionDay + 1));
  const delta = subtract(next, prev);

  const kmPerAu = 149_597_870.7;
  const secondsPerDay = 86_400;

  return {
    x: (delta.x * kmPerAu) / (2 * secondsPerDay),
    y: (delta.y * kmPerAu) / (2 * secondsPerDay),
    z: (delta.z * kmPerAu) / (2 * secondsPerDay),
  };
}

function computeBaseEvents(day: number): MissionEvent[] {
  const timeline: MissionEvent[] = [
    {
      id: "launch-window",
      missionDay: 0,
      title: "Launch Window Open",
      detail: "Earth departure burn constraints validated against nominal C3 envelope.",
      severity: "info",
    },
    {
      id: "deep-space-maneuver",
      missionDay: 124,
      title: "Deep Space Maneuver",
      detail: "Trajectory correction burn aligns B-plane for Jovian arrival corridor.",
      severity: "info",
    },
    {
      id: "belt-crossing",
      missionDay: 238,
      title: "Asteroid Belt Transit",
      detail: "Elevated conjunction checks active across tracked main-belt objects.",
      severity: "warning",
    },
    {
      id: "jupiter-soi",
      missionDay: 588,
      title: "Jovian SOI Entry",
      detail: "Jupiter sphere-of-influence transition; high-fidelity gravity model activated.",
      severity: "info",
    },
    {
      id: "capture-burn",
      missionDay: 636,
      title: "Jupiter Capture Burn",
      detail: "Primary capture maneuver executed near perijove with propellant margin preserved.",
      severity: "critical",
    },
  ];

  return timeline.filter((event) => event.missionDay <= day);
}

class MockSimulationApi implements SimulationApi {
  private missionDay = 0;
  private readonly trajectory = generateTrajectoryPoints();
  private dynamicEvents: MissionEvent[] = [];
  private promptIndex = 0;
  private target = "Jupiter";
  private guidanceMode: "auto" | "manual" = "auto";
  private deltaVBias = 0;

  async bootstrap(): Promise<SimulationSnapshot> {
    return this.snapshot();
  }

  async advance(deltaMissionDays: number): Promise<SimulationSnapshot> {
    this.missionDay = clamp(this.missionDay + deltaMissionDays, 0, MAX_DAY);
    return this.snapshot();
  }

  async seek(missionDay: number): Promise<SimulationSnapshot> {
    this.missionDay = clamp(missionDay, 0, MAX_DAY);
    return this.snapshot();
  }

  async sendPrompt(command: PromptCommand): Promise<PromptResult> {
    const text = command.prompt.toLowerCase();
    const createdEvents: MissionEvent[] = [];

    if (text.includes("manual")) {
      this.guidanceMode = "manual";
      createdEvents.push(this.newEvent("Manual Guidance Enabled", "Prompt commanded manual thrust authority."));
    }

    if (text.includes("auto")) {
      this.guidanceMode = "auto";
      createdEvents.push(this.newEvent("Autopilot Restored", "Flight software resumed closed-loop guidance."));
    }

    if (text.includes("europa")) {
      this.target = "Europa";
      createdEvents.push(this.newEvent("Target Updated", "Mission branch changed to Europa science orbit insertion."));
    } else if (text.includes("ganymede")) {
      this.target = "Ganymede";
      createdEvents.push(this.newEvent("Target Updated", "Mission branch changed to Ganymede flyby campaign."));
    } else if (text.includes("callisto")) {
      this.target = "Callisto";
      createdEvents.push(this.newEvent("Target Updated", "Mission branch changed to Callisto rendezvous geometry."));
    } else if (text.includes("jupiter")) {
      this.target = "Jupiter";
    }

    if (text.includes("burn") || text.includes("delta-v") || text.includes("thrust")) {
      this.deltaVBias += 0.12;
      createdEvents.push(this.newEvent("Burn Directive", "Guidance accepted a finite burn update from prompt plan."));
    }

    if (text.includes("fast") || text.includes("accelerate")) {
      this.missionDay = clamp(this.missionDay + 12, 0, MAX_DAY);
      createdEvents.push(this.newEvent("Time Compression", "Simulation advanced to evaluate prompt what-if maneuver."));
    }

    if (createdEvents.length === 0) {
      createdEvents.push(
        this.newEvent(
          "Prompt Parsed",
          "Mission assistant acknowledged request and queued no-op for current timeline.",
        ),
      );
    }

    this.dynamicEvents = [...this.dynamicEvents, ...createdEvents].slice(-16);

    const assistant = [
      `Guidance mode: ${this.guidanceMode.toUpperCase()}`,
      `Active target: ${this.target}`,
      "Trajectory state updated in mock mission core.",
    ].join(" | ");

    return {
      assistant,
      createdEvents,
    };
  }

  private newEvent(title: string, detail: string): MissionEvent {
    this.promptIndex += 1;
    return {
      id: `prompt-${this.promptIndex}`,
      missionDay: Math.round(this.missionDay),
      title,
      detail,
      severity: "info",
    };
  }

  private snapshot(): SimulationSnapshot {
    const bodies = buildBodyStates(this.missionDay);
    const craftPosition = samplePath(this.trajectory, this.missionDay);
    const craftVelocity = estimateVelocity(this.trajectory, this.missionDay);
    const speed = vectorMagnitude(craftVelocity) + this.deltaVBias;

    const events = [...computeBaseEvents(this.missionDay), ...this.dynamicEvents]
      .sort((a, b) => a.missionDay - b.missionDay)
      .slice(-10);

    return {
      sessionId: "mock-jovian-session",
      missionName: "Asteria Jupiter Survey",
      missionDay: this.missionDay,
      simTimeUtc: new Date(Date.UTC(2034, 5, 14 + Math.floor(this.missionDay))).toISOString(),
      phase: formatMissionPhase(this.missionDay),
      spacecraft: {
        name: "Asteria-1",
        positionAu: craftPosition,
        velocityKmS: craftVelocity,
        heliocentricSpeedKmS: speed,
        propellantRemainingPct: clamp(100 - (this.missionDay / MAX_DAY) * 67 - this.deltaVBias * 2.2, 18, 100),
        deltaVUsedKmS: Number((this.missionDay / MAX_DAY * 10.8 + this.deltaVBias).toFixed(3)),
        target: this.target,
        guidanceMode: this.guidanceMode,
      },
      bodies,
      trajectoryAu: this.trajectory,
      events,
    };
  }
}

let singleton: MockSimulationApi | null = null;

export function createMockSimulationApi(): SimulationApi {
  if (!singleton) {
    singleton = new MockSimulationApi();
  }
  return singleton;
}
