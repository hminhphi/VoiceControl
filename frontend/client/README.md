# Orchestrator Frontend

This is the frontend for the Multi-Agent Orchestrator, built with React, Vite, and Tailwind CSS.

## Installation

1. Open a terminal and navigate to this folder:
   ```sh
   cd frontend/client
   ```
2. Install dependencies:
   ```sh
   npm install
   ```
3. Start the development server:
   ```sh
   npm run dev
   ```
   The app will be available at http://localhost:5173 (or as shown in your terminal).

## Project Structure
- `src/` — Main source code (React components, services, styles)
- `public/` — Static assets
- `package.json` — Project dependencies and scripts
- `tailwind.config.cjs` — Tailwind CSS configuration
- `vite.config.js` — Vite configuration

## Features
- Agent management (add/remove agents, sync with backend)
- Live system log (WebSocket events)
- Interactive canvas for agent visualization
- Responsive layout with resizable panels

## Backend Requirements
- REST API endpoints for agent management:
  - `POST /api/agents` — Add agent
  - `DELETE /api/agents/{agent_id}` — Remove agent
- WebSocket endpoint at `/ws` for live updates (optional)

## Build for Production
```sh
npm run build
```

## License
MIT
