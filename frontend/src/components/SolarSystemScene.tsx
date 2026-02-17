import { useEffect, useMemo, useRef } from "react";
import {
  EphemerisTable,
  Simulation as SpacekitSimulation,
  SpaceObjectPresets,
  type SpaceObject,
} from "spacekit.js";

import type { SimulationSnapshot } from "../api/contracts";

const MAX_MISSION_DAY = 760;
const DAY_MS = 86_400_000;
const GALILEAN_NAMES = new Set(["io", "europa", "ganymede", "callisto"]);

interface SpacekitResources {
  sim: SpacekitSimulation;
  objects: Record<string, SpaceObject>;
  spacecraft: SpaceObject | null;
}

interface FocusTarget {
  key: string;
  target: SpaceObject | null;
  offset: [number, number, number];
}

function toJulianDate(date: Date): number {
  return date.getTime() / DAY_MS + 2_440_587.5;
}

function getMissionZeroDate(snapshot: SimulationSnapshot): Date {
  const current = new Date(snapshot.simTimeUtc);
  return new Date(current.getTime() - snapshot.missionDay * DAY_MS);
}

function missionDayFromDate(date: Date, missionZeroDate: Date): number {
  return (date.getTime() - missionZeroDate.getTime()) / DAY_MS;
}

function buildTrajectoryEphemeris(snapshot: SimulationSnapshot): EphemerisTable {
  const points = snapshot.trajectoryAu;
  const rows: number[][] = [];

  const missionStartJd = toJulianDate(getMissionZeroDate(snapshot));
  const dayStep = points.length > 1 ? MAX_MISSION_DAY / (points.length - 1) : 1;

  for (let i = 0; i < points.length; i += 1) {
    const point = points[i];
    const left = points[Math.max(i - 1, 0)];
    const right = points[Math.min(i + 1, points.length - 1)];

    const divisor = i === 0 || i === points.length - 1 ? dayStep : dayStep * 2;

    rows.push([
      missionStartJd + i * dayStep,
      point.x,
      point.y,
      point.z,
      (right.x - left.x) / divisor,
      (right.y - left.y) / divisor,
      (right.z - left.z) / divisor,
    ]);
  }

  return new EphemerisTable({
    data: rows,
    ephemerisType: "cartesianposvel",
    interpolationType: "lagrange",
    interpolationOrder: 5,
    distanceUnits: "au",
    timeUnits: "day",
  });
}

function getFocusTarget(snapshot: SimulationSnapshot, resources: SpacekitResources): FocusTarget {
  const jovianPhase =
    snapshot.phase === "jovian_approach" ||
    snapshot.phase === "jovian_capture" ||
    snapshot.phase === "science";

  if (jovianPhase) {
    if (resources.objects.jupiter) {
      return {
        key: "jupiter",
        target: resources.objects.jupiter,
        offset: [-4.6, -5.2, 2.1],
      };
    }
    return {
      key: "spacecraft",
      target: resources.spacecraft,
      offset: [-2.8, -2.8, 1.1],
    };
  }

  if (resources.spacecraft) {
    return {
      key: "spacecraft",
      target: resources.spacecraft,
      offset: [-2.8, -2.8, 1.1],
    };
  }

  return {
    key: "earth",
    target: resources.objects.earth ?? null,
    offset: [-2.2, -2.2, 0.9],
  };
}

function applyFocus(snapshot: SimulationSnapshot, resources: SpacekitResources): string | null {
  const viewer = resources.sim.getViewer();
  const focus = getFocusTarget(snapshot, resources);
  const target = focus.target;

  if (!target) {
    return null;
  }

  const anchor = target.get3jsObjects()[0];
  if (!anchor) {
    return null;
  }

  // One-shot recenter only: continuous follow causes OrbitControls zoom lag.
  viewer.stopFollowingObject();
  const camera = viewer.get3jsCamera();
  const controls = viewer.get3jsCameraControls();
  const [ox, oy, oz] = focus.offset;

  camera.position.set(
    anchor.position.x + ox,
    anchor.position.y + oy,
    anchor.position.z + oz,
  );
  controls.target.set(anchor.position.x, anchor.position.y, anchor.position.z);
  controls.update();

  return focus.key;
}

function createSceneObjects(sim: SpacekitSimulation, snapshot: SimulationSnapshot): SpacekitResources {
  const objects: Record<string, SpaceObject> = {};

  const sun = sim.createObject("sun", {
    ...SpaceObjectPresets.SUN,
    particleSize: 220,
  });
  objects.sun = sun;

  const mercury = sim.createObject("mercury", {
    ...SpaceObjectPresets.MERCURY,
    particleSize: 16,
    hideOrbit: true,
  });
  objects.mercury = mercury;

  const venus = sim.createObject("venus", {
    ...SpaceObjectPresets.VENUS,
    particleSize: 18,
    hideOrbit: true,
  });
  objects.venus = venus;

  const earth = sim.createSphere("earth", {
    ephem: SpaceObjectPresets.EARTH.ephem,
    labelText: "Earth",
    textureUrl:
      "https://raw.githubusercontent.com/typpo/spacekit/master/examples/basic_asteroid_earth_flyby/earthtexture.jpg",
    radius: 0.0042,
    levelsOfDetail: [
      { radii: 0, segments: 40 },
      { radii: 80, segments: 20 },
      { radii: 160, segments: 10 },
    ],
    atmosphere: {
      enable: true,
      color: 0x7ab2ff,
    },
    hideOrbit: true,
  });
  objects.earth = earth;

  const mars = sim.createObject("mars", {
    ...SpaceObjectPresets.MARS,
    particleSize: 15,
    hideOrbit: true,
  });
  objects.mars = mars;

  const jupiter = sim.createSphere("jupiter", {
    ...SpaceObjectPresets.JUPITER,
    labelText: "Jupiter",
    textureUrl:
      "https://raw.githubusercontent.com/typpo/spacekit/master/examples/jupiter/jupiter2_4k.jpg",
    radius: 0.017,
    atmosphere: {
      enable: true,
      color: 0xc7c1a8,
    },
    levelsOfDetail: [
      { radii: 0, segments: 64 },
      { radii: 120, segments: 32 },
      { radii: 220, segments: 16 },
    ],
    rotation: {
      enable: true,
      speed: 1.8,
      period: 0.41,
    },
    hideOrbit: true,
  });
  objects.jupiter = jupiter;

  const saturn = sim.createSphere("saturn", {
    ephem: SpaceObjectPresets.SATURN.ephem,
    textureUrl:
      "https://raw.githubusercontent.com/typpo/spacekit/master/examples/saturn/th_saturn.png",
    radius: 0.013,
    levelsOfDetail: [
      { radii: 0, segments: 36 },
      { radii: 90, segments: 18 },
      { radii: 180, segments: 10 },
    ],
    hideOrbit: true,
  });
  objects.saturn = saturn;

  const asteroidPoints = snapshot.bodies
    .filter((body) => body.kind === "asteroid")
    .map((body) => [body.positionAu.x, body.positionAu.y, body.positionAu.z] as [number, number, number]);

  if (asteroidPoints.length > 0) {
    sim.createStaticParticles("main-belt", asteroidPoints, {
      defaultColor: 0x8f9198,
      size: 1.4,
    });
  }

  let spacecraft: SpaceObject | null = null;
  if (snapshot.trajectoryAu.length > 1) {
    spacecraft = sim.createObject(snapshot.spacecraft.name, {
      ephemTable: buildTrajectoryEphemeris(snapshot),
      labelText: snapshot.spacecraft.name,
      textureUrl: "{{assets}}/sprites/lensflare0.png",
      scale: [0.28, 0.28, 0.28],
      particleSize: 90,
      theme: {
        color: 0x6bf4ff,
        orbitColor: 0x6bf4ff,
      },
      orbitPathSettings: {
        leadDurationYears: 2.4,
        trailDurationYears: 0.35,
        numberSamplePoints: 220,
      },
      ecliptic: {
        displayLines: false,
      },
    });
  }

  sim.loadNaturalSatellites()
    .then((loader) => {
      const moons = loader
        .getSatellitesForPlanet("jupiter")
        .filter((moon: { name: string }) => GALILEAN_NAMES.has(moon.name.toLowerCase()));

      moons.forEach((moon: { name: string; ephem: unknown }) => {
        sim.createObject(`jupiter-${moon.name.toLowerCase()}`, {
          ephem: moon.ephem as any,
          particleSize: 22,
          theme: {
            color: 0xf7deab,
            orbitColor: 0x4c5570,
          },
          hideOrbit: true,
        });
      });
    })
    .catch(() => {
      // Ignore optional moon-loader failures in mock/degraded mode.
    });

  return { sim, objects, spacecraft };
}

function updateTimeFlow(
  resources: SpacekitResources,
  snapshot: SimulationSnapshot,
  daysPerSecond: number,
  missionZeroDate: Date,
  forceSyncDate = false,
): void {
  const simMissionDay = missionDayFromDate(resources.sim.getDate(), missionZeroDate);
  if (forceSyncDate || Math.abs(simMissionDay - snapshot.missionDay) > 0.9) {
    resources.sim.setDate(new Date(snapshot.simTimeUtc));
  }

  if (daysPerSecond <= 0) {
    resources.sim.stop();
  } else {
    resources.sim.setJdPerSecond(daysPerSecond);
    resources.sim.start();
  }
}

export default function SolarSystemScene({
  snapshot,
  daysPerSecond,
}: {
  snapshot: SimulationSnapshot | null;
  daysPerSecond: number;
}): JSX.Element {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const resourcesRef = useRef<SpacekitResources | null>(null);
  const focusKeyRef = useRef<string | null>(null);
  const missionZeroDateRef = useRef<Date | null>(null);

  const targetLabel = useMemo(() => snapshot?.spacecraft.target ?? "-", [snapshot?.spacecraft.target]);

  useEffect(() => {
    if (!containerRef.current || !snapshot || resourcesRef.current) {
      return;
    }

    const sim = new SpacekitSimulation(containerRef.current as unknown as HTMLCanvasElement, {
      basePath: "https://typpo.github.io/spacekit/src",
      startDate: new Date(snapshot.simTimeUtc),
      jdPerSecond: Math.max(daysPerSecond, 0.2),
      startPaused: false,
      unitsPerAu: 12,
      particleTextureUrl: "{{assets}}/sprites/fuzzyparticle-circled.png",
      camera: {
        enableDrift: false,
        initialPosition: [0, -94, 38],
      },
      debug: {
        showAxes: false,
        showGrid: false,
        showStats: false,
      },
    });

    const renderer = sim.getRenderer();
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.0));

    const viewer = sim.getViewer();
    const controls = viewer.get3jsCameraControls();
    controls.minDistance = 0.0015;
    controls.maxDistance = 600;
    controls.zoomSpeed = 2.8;
    controls.panSpeed = 1.6;
    controls.rotateSpeed = 1.6;
    controls.dampingFactor = 0.08;

    const camera = viewer.get3jsCamera();
    camera.near = 0.0000008;
    camera.far = 4000;
    camera.updateProjectionMatrix();

    sim.createAmbientLight(0x12192a);
    sim.createLight([0, 0, 0], 0xffffff);
    sim.createStars({ minSize: 0.8 });

    missionZeroDateRef.current = getMissionZeroDate(snapshot);
    resourcesRef.current = createSceneObjects(sim, snapshot);
    updateTimeFlow(
      resourcesRef.current,
      snapshot,
      daysPerSecond,
      missionZeroDateRef.current,
      true,
    );
    focusKeyRef.current = applyFocus(snapshot, resourcesRef.current);

    return () => {
      if (resourcesRef.current) {
        resourcesRef.current.sim.stop();
      }
      resourcesRef.current = null;
      focusKeyRef.current = null;
      missionZeroDateRef.current = null;
      if (containerRef.current) {
        containerRef.current.innerHTML = "";
      }
    };
  }, [snapshot?.sessionId]);

  useEffect(() => {
    if (!snapshot || !resourcesRef.current || !missionZeroDateRef.current) {
      return;
    }

    const resources = resourcesRef.current;
    updateTimeFlow(
      resources,
      snapshot,
      daysPerSecond,
      missionZeroDateRef.current,
      false,
    );

    const nextFocusKey = getFocusTarget(snapshot, resources).key;
    if (focusKeyRef.current !== nextFocusKey) {
      focusKeyRef.current = applyFocus(snapshot, resources);
    }
  }, [daysPerSecond, snapshot]);

  return (
    <div className="scene-shell">
      <div ref={containerRef} className="spacekit-root" />
      {snapshot ? (
        <div className="scene-hud">
          <p>Target: {targetLabel}</p>
          <p>Mode: {snapshot.spacecraft.guidanceMode.toUpperCase()}</p>
          <p>Bodies: {snapshot.bodies.length}</p>
        </div>
      ) : null}
    </div>
  );
}
