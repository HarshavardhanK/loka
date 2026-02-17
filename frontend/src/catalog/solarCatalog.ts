import type { BodyState, Vector3 } from "../api/contracts";

const AU_KM = 149_597_870.7;
const ASTEROID_COUNT = 140;

interface BodyDefinition {
  id: string;
  name: string;
  kind: BodyState["kind"];
  radiusKm: number;
  color: string;
  parentId: string | null;
  orbitRadiusAu: number;
  orbitalPeriodDays: number;
  phaseOffsetRad: number;
}

interface AsteroidDefinition {
  id: string;
  radiusKm: number;
  color: string;
  semiMajorAu: number;
  eccentricity: number;
  orbitalPeriodDays: number;
  inclinationRad: number;
  phaseOffsetRad: number;
}

const MAJOR_BODIES: BodyDefinition[] = [
  {
    id: "sun",
    name: "Sun",
    kind: "star",
    radiusKm: 696_340,
    color: "#fcb65f",
    parentId: null,
    orbitRadiusAu: 0,
    orbitalPeriodDays: 0,
    phaseOffsetRad: 0,
  },
  {
    id: "mercury",
    name: "Mercury",
    kind: "planet",
    radiusKm: 2_439.7,
    color: "#9ea3a9",
    parentId: "sun",
    orbitRadiusAu: 0.387,
    orbitalPeriodDays: 88,
    phaseOffsetRad: 0.8,
  },
  {
    id: "venus",
    name: "Venus",
    kind: "planet",
    radiusKm: 6_051.8,
    color: "#e6c39c",
    parentId: "sun",
    orbitRadiusAu: 0.723,
    orbitalPeriodDays: 224.7,
    phaseOffsetRad: 1.2,
  },
  {
    id: "earth",
    name: "Earth",
    kind: "planet",
    radiusKm: 6_371,
    color: "#3f9dff",
    parentId: "sun",
    orbitRadiusAu: 1,
    orbitalPeriodDays: 365.256,
    phaseOffsetRad: 2.4,
  },
  {
    id: "moon",
    name: "Moon",
    kind: "moon",
    radiusKm: 1_737.4,
    color: "#c4c8d1",
    parentId: "earth",
    orbitRadiusAu: 384_400 / AU_KM,
    orbitalPeriodDays: 27.3,
    phaseOffsetRad: 0.2,
  },
  {
    id: "mars",
    name: "Mars",
    kind: "planet",
    radiusKm: 3_389.5,
    color: "#d07254",
    parentId: "sun",
    orbitRadiusAu: 1.524,
    orbitalPeriodDays: 687,
    phaseOffsetRad: -0.4,
  },
  {
    id: "jupiter",
    name: "Jupiter",
    kind: "planet",
    radiusKm: 69_911,
    color: "#d5a979",
    parentId: "sun",
    orbitRadiusAu: 5.204,
    orbitalPeriodDays: 4_332.59,
    phaseOffsetRad: -1.6,
  },
  {
    id: "io",
    name: "Io",
    kind: "moon",
    radiusKm: 1_821.6,
    color: "#f2d2a2",
    parentId: "jupiter",
    orbitRadiusAu: 421_700 / AU_KM,
    orbitalPeriodDays: 1.769,
    phaseOffsetRad: 0.7,
  },
  {
    id: "europa",
    name: "Europa",
    kind: "moon",
    radiusKm: 1_560.8,
    color: "#dfd8cb",
    parentId: "jupiter",
    orbitRadiusAu: 671_100 / AU_KM,
    orbitalPeriodDays: 3.551,
    phaseOffsetRad: 1.4,
  },
  {
    id: "ganymede",
    name: "Ganymede",
    kind: "moon",
    radiusKm: 2_634.1,
    color: "#b7aa97",
    parentId: "jupiter",
    orbitRadiusAu: 1_070_400 / AU_KM,
    orbitalPeriodDays: 7.154,
    phaseOffsetRad: 2.1,
  },
  {
    id: "callisto",
    name: "Callisto",
    kind: "moon",
    radiusKm: 2_410.3,
    color: "#8f8a84",
    parentId: "jupiter",
    orbitRadiusAu: 1_882_700 / AU_KM,
    orbitalPeriodDays: 16.689,
    phaseOffsetRad: 0.3,
  },
  {
    id: "saturn",
    name: "Saturn",
    kind: "planet",
    radiusKm: 58_232,
    color: "#ccb98d",
    parentId: "sun",
    orbitRadiusAu: 9.58,
    orbitalPeriodDays: 10_759,
    phaseOffsetRad: 0.8,
  },
];

function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let n = Math.imul(t ^ (t >>> 15), t | 1);
    n ^= n + Math.imul(n ^ (n >>> 7), n | 61);
    return ((n ^ (n >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function buildAsteroidDefinitions(count: number): AsteroidDefinition[] {
  const rand = mulberry32(42_198_112);
  const definitions: AsteroidDefinition[] = [];
  const colors = ["#9d9b96", "#b0aca4", "#8e8a86", "#7d7f82"];

  for (let i = 0; i < count; i += 1) {
    const semiMajorAu = 2.1 + rand() * 1.2;
    const eccentricity = 0.01 + rand() * 0.17;
    const periodDays = Math.sqrt(semiMajorAu ** 3) * 365.25;
    const inclination = (rand() - 0.5) * 0.22;
    definitions.push({
      id: `asteroid-${i + 1}`,
      radiusKm: 3 + rand() * 65,
      color: colors[Math.floor(rand() * colors.length)],
      semiMajorAu,
      eccentricity,
      orbitalPeriodDays: periodDays,
      inclinationRad: inclination,
      phaseOffsetRad: rand() * Math.PI * 2,
    });
  }

  return definitions;
}

const ASTEROIDS = buildAsteroidDefinitions(ASTEROID_COUNT);

function circularPosition(radiusAu: number, periodDays: number, day: number, phase: number): Vector3 {
  if (radiusAu === 0 || periodDays <= 0) {
    return { x: 0, y: 0, z: 0 };
  }

  const angle = phase + (day / periodDays) * Math.PI * 2;
  return {
    x: radiusAu * Math.cos(angle),
    y: 0,
    z: radiusAu * Math.sin(angle),
  };
}

function ellipticalPosition(
  semiMajorAu: number,
  eccentricity: number,
  periodDays: number,
  day: number,
  phase: number,
  inclinationRad: number,
): Vector3 {
  const angle = phase + (day / periodDays) * Math.PI * 2;
  const radius = (semiMajorAu * (1 - eccentricity ** 2)) / (1 + eccentricity * Math.cos(angle));

  return {
    x: radius * Math.cos(angle),
    y: radius * Math.sin(inclinationRad) * 0.18,
    z: radius * Math.sin(angle),
  };
}

function velocityApprox(position: Vector3, radiusAu: number, periodDays: number): Vector3 {
  if (radiusAu === 0 || periodDays <= 0) {
    return { x: 0, y: 0, z: 0 };
  }

  const omega = (Math.PI * 2) / (periodDays * 86_400);

  return {
    x: -position.z * omega * AU_KM,
    y: 0,
    z: position.x * omega * AU_KM,
  };
}

function bodyFromDefinition(def: BodyDefinition, missionDay: number, parent: BodyState | undefined): BodyState {
  const localPosition = circularPosition(
    def.orbitRadiusAu,
    def.orbitalPeriodDays,
    missionDay,
    def.phaseOffsetRad,
  );

  const worldPosition: Vector3 = parent
    ? {
        x: parent.positionAu.x + localPosition.x,
        y: parent.positionAu.y + localPosition.y,
        z: parent.positionAu.z + localPosition.z,
      }
    : localPosition;

  return {
    id: def.id,
    name: def.name,
    kind: def.kind,
    radiusKm: def.radiusKm,
    color: def.color,
    parentId: def.parentId,
    orbitRadiusAu: def.orbitRadiusAu,
    orbitalPeriodDays: def.orbitalPeriodDays,
    positionAu: worldPosition,
    velocityKmS: velocityApprox(localPosition, def.orbitRadiusAu, def.orbitalPeriodDays),
  };
}

function asteroidState(def: AsteroidDefinition, missionDay: number): BodyState {
  const position = ellipticalPosition(
    def.semiMajorAu,
    def.eccentricity,
    def.orbitalPeriodDays,
    missionDay,
    def.phaseOffsetRad,
    def.inclinationRad,
  );

  return {
    id: def.id,
    name: def.id,
    kind: "asteroid",
    radiusKm: def.radiusKm,
    color: def.color,
    parentId: "sun",
    orbitRadiusAu: def.semiMajorAu,
    orbitalPeriodDays: def.orbitalPeriodDays,
    positionAu: position,
    velocityKmS: velocityApprox(position, def.semiMajorAu, def.orbitalPeriodDays),
  };
}

export function buildBodyStates(missionDay: number): BodyState[] {
  const byId = new Map<string, BodyState>();

  for (const def of MAJOR_BODIES) {
    const parent = def.parentId ? byId.get(def.parentId) : undefined;
    byId.set(def.id, bodyFromDefinition(def, missionDay, parent));
  }

  const asteroidStates = ASTEROIDS.map((asteroid) => asteroidState(asteroid, missionDay));

  return [...byId.values(), ...asteroidStates];
}

export function getBodyById(bodies: BodyState[], id: string): BodyState | undefined {
  return bodies.find((body) => body.id === id);
}

export function formatMissionPhase(day: number):
  | "launch"
  | "escape"
  | "cruise"
  | "jovian_approach"
  | "jovian_capture"
  | "science" {
  if (day < 12) {
    return "launch";
  }
  if (day < 45) {
    return "escape";
  }
  if (day < 470) {
    return "cruise";
  }
  if (day < 590) {
    return "jovian_approach";
  }
  if (day < 670) {
    return "jovian_capture";
  }
  return "science";
}

export const AU_IN_KM = AU_KM;
