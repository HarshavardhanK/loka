# Loka Mission Control Frontend

Web-native simulation UI for prompt-driven solar-system navigation.

## Stack
- React + TypeScript + Vite
- Spacekit.js (Three.js-based orbital visualization engine)
- Typed simulation API adapter with mock/live switch

## Run
```bash
cd frontend
npm install
npm run dev
```

## Modes
Default mode is mock:
- `VITE_SIMULATION_MODE=mock` (or unset)

Live API mode:
```bash
VITE_SIMULATION_MODE=live VITE_SIMULATION_API_BASE_URL=http://localhost:8000 npm run dev
```

Expected endpoints in live mode:
- `POST /simulation/bootstrap`
- `POST /simulation/advance`
- `POST /simulation/seek`
- `POST /simulation/prompt`
