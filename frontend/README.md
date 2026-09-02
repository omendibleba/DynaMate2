# DynaMate2 frontend

React + TypeScript + Vite + Tailwind CSS UI for DynaMate2, talking to the FastAPI
backend in `../backend/`.

## Development

```bash
# from the frontend/ directory, with Node.js on PATH
# (see /groups/ycolon/group-envs/agentic-tutorials — this repo's working environment)
npm install       # first time only
npm run dev       # Vite dev server on :5173, proxying /api -> localhost:8000
```

Run the backend separately (see `../backend/main.py` / the repo root README) on port
8000 — the Vite dev proxy (`vite.config.ts`) forwards `/api/*` requests to it so the
browser sees everything as same-origin.

## Production build

```bash
npm run build     # tsc -b && vite build -> dist/
```

`backend/main.py` serves `dist/` as static files once built, so `python server.py`
from the repo root becomes the single-command entry point (matching the old
`python app.py` Gradio UI, preserved on the `gradio-ui-legacy` branch).

## Structure

- `src/lib/api.ts` — typed client for the backend API, including the hand-rolled
  SSE parser for the streaming chat endpoint (browser `EventSource` can't send a
  POST body, so this uses `fetch` + `ReadableStream` instead).
- `src/hooks/useChatStream.ts` — drives one chat turn end-to-end.
- `src/components/` — `ChatPanel`, `QuickStartPanel` (mirrors the tutorial-notebook
  quick-start prompts), `FileUploadZone`, `StatusSidebar`, `ThreadHistory`,
  `AgentTracePanel`.

## Checks

```bash
npm run build     # type-checks (tsc -b) and production-builds
npm run lint       # oxlint
```
