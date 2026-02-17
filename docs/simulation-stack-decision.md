# Loka Simulation Stack Decision (Feb 16, 2026)

## Decision Summary
Use a **web-native runtime** for the product surface and a **physics service layer** based on actively maintained astrodynamics libraries.

Chosen path:
- Frontend: `React + Spacekit.js` (with typed API adapter so scene engine can be swapped later if needed)
- Mission API layer: `FastAPI` (Python orchestrator)
- High-fidelity trajectory core: `Orekit` service (Java, Apache-2.0)
- Ephemeris/geometry kernels: `SPICE` via `SpiceyPy` (MIT)
- RL inference control: vLLM/cluster service behind mission-planning API

## Why This Path
1. Supports prompt-driven mission UX directly in browser with no desktop embed constraints.
2. Keeps physics in dedicated services, avoiding hand-rolled "vibe" orbital mechanics.
3. Uses active OSS projects with permissive licenses aligned to this repo.
4. Gives a clean transition from mocked UI flow to production backend endpoints.
5. Reuses mature open-source orbital rendering primitives (Spacekit) instead of custom scene math.

## Why Not Celestia Runtime
Pros:
- Great rendering quality and active maintenance.
- Rich object catalogs and scripting support.

Cons for this product:
- Desktop-first app model, not browser-native.
- No first-class API contract for prompt-driven mission orchestration.
- GPL-2.0 implications for tight integration/distribution.

Use Celestia as a **visual benchmark and optional validation viewer**, not core runtime.

## Spacekit Role
Pros:
- Fastest path to a realistic browser simulation scene with ephemerides and orbital controls.
- Supports natural satellites, ephemeris tables, and camera-follow behavior needed for Earth→Jupiter flows.

Limitations:
- Rendering layer only; not a mission-ops physics/planning backend.
- Older dependency line means we keep the simulation API contract decoupled from the renderer.

Decision:
- Use Spacekit for the frontend scene runtime now.
- Keep all mission-state/trajectory logic behind typed backend APIs so backend realism is independent of renderer.

## Backend Layering
## 1) API Gateway (`FastAPI`)
- Auth/session management.
- Mission timeline state store.
- Orchestration across inference + trajectory + ephemeris services.
- Server-sent events/WebSocket for streaming updates.

## 2) Guidance/Planning Service (`LLM`)
- Accept operator prompt and mission context.
- Produce structured command plan (`burn`, `retarget`, `time step`, `constraints`).
- Validate command schema before dispatch.

## 3) Trajectory Service (`Orekit`)
- Lambert solves, patched conic sequences, capture/insertion profiles.
- Event detection (SOI entry, periapsis, eclipse windows).
- Maneuver optimization with constraints.

## 4) Ephemeris Service (`SpiceyPy`)
- Body states from NAIF kernels.
- Frame transformations and time standards.
- Support for planets, moons, and selected small bodies.

## 5) Simulation State Store
- Immutable per-step snapshots.
- Replay and deterministic seed controls.
- Telemetry stream payloads for frontend scene.

## API Contract (Frontend Already Uses)
Current frontend contract:
- `POST /simulation/bootstrap`
- `POST /simulation/advance` `{ deltaMissionDays }`
- `POST /simulation/seek` `{ missionDay }`
- `POST /simulation/prompt` `{ prompt }`

Response model includes:
- Body states (planets, moons, asteroids)
- Spacecraft state
- Mission phase/events
- Trajectory polyline

## Execution Plan
1. Keep frontend in mock mode for rapid UX iteration.
2. Implement FastAPI endpoints matching current contract.
3. Add Orekit + SPICE adapters under those endpoints.
4. Wire prompt planner to cluster inference.
5. Add integration tests for Earth-to-Jupiter scenarios.
