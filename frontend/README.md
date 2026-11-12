# DermaGuard Frontend

React + Vite frontend for the DermaGuard multi-agent clinical triage system.

## Prerequisites

- Node.js 18+
- DermaGuard FastAPI backend running on `http://127.0.0.1:8000`

## Start the Backend First

```bash
# In the DermaGuard root directory:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

## Features

- **Drag-and-drop image upload** — JPEG, PNG, WebP
- **Patient parameter inputs** — age, biological sex, anatomical site
- **Grad-CAM colormap selector** — Jet (clinical) or Turbo (high contrast)
- **Decision banner** — `AUTO_TRIAGE` / `REFER` / `RE_IMAGE` with urgency level
- **Grad-CAM heatmap viewer** — 3-view switcher: overlay, original, pure heatmap
- **Per-agent result cards** — all 6 agents with metrics, progress bars, and flags
- **Gemini clinical explanation** — natural language summary
- **Raw JSON viewer** — full pipeline response

## Build for Production

```bash
npm run build
# Output in dist/
```
