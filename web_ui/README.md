# CORTEX Dashboard

A modern, dark-theme web UI for the [CORTEX](https://github.com/your-org/CORTEX) RAG/AI assistant. Built with React, TypeScript, Vite, Framer Motion, and Recharts.

## Features

- **Dark theme** with soft contrast, glassmorphism, and semantic color tokens
- **Responsive layout**: collapsible sidebar, top bar with search, notifications, voice command
- **Pages**: Overview (stats + charts), Search, Documents (grid/list + hover previews), Chat (typed messages), Settings (toggles, profile)
- **Micro-animations**: Framer Motion on cards, buttons, lists, and chart tooltips
- **Accessibility**: Skip link, landmarks, ARIA, focus-visible, reduced-motion support
- **Optional voice**: Web Speech API for voice search/commands (browser-dependent)

## Quick Start

```bash
cd web_ui
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Scripts

| Command    | Description        |
|-----------|--------------------|
| `npm run dev`    | Start dev server (port 5173) |
| `npm run build`  | Production build   |
| `npm run preview`| Preview production build |

## Project Structure

```
web_ui/
├── DESIGN_SPEC.md      # UI design specification & style guide
├── ACCESSIBILITY.md    # Accessibility notes & performance
├── src/
│   ├── theme/         # CSS design tokens (tokens.css)
│   ├── components/
│   │   ├── layout/     # Sidebar, TopBar, Layout
│   │   ├── ui/         # Button, Card, inputs, toggle
│   │   ├── charts/     # Recharts wrappers + tooltips
│   │   └── voice/      # VoiceCommand module
│   ├── pages/         # Home, Search, Documents, Chat, Settings
│   └── data/          # Placeholder data
└── public/
```

## Design

See [DESIGN_SPEC.md](./DESIGN_SPEC.md) for:

- Color palette and glassmorphism tokens
- Typography scale and kinetic type notes
- Component specs (sidebar, cards, buttons, charts, chat)
- Motion timing and reduced-motion behavior
- Responsive breakpoints

## Backend Integration

This UI uses placeholder data. To connect to the CORTEX Python backend:

1. Add an API client (e.g. `fetch` or axios) in `src/api/`.
2. Replace imports from `src/data/placeholder.ts` with API calls.
3. Wire Search to your search/RAG endpoint, Chat to your streaming chat endpoint, and Documents to your document list endpoint.
4. Optionally add WebSocket or SSE for real-time stats on the Overview page.

## License

Same as the parent CORTEX project.
