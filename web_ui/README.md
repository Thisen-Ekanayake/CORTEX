# Cortex — Enterprise Intelligence Console

Enterprise UI for the Cortex on-premise RAG-based AI system. Three-panel layout: Top Bar, Left Sidebar (workspaces, query log, data sources), Main Intelligence Panel, Evidence Panel, Bottom Query Input Bar.

## Run locally

```bash
cd web_ui
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Build

```bash
npm run build
npm run preview
```

## Folder structure

```
src/
  App.tsx
  main.tsx
  index.css
  types/
    index.ts
  data/
    mockData.ts
  components/
    layout/
      AppLayout.tsx
    TopBar/
      TopBar.tsx
      WorkspaceSelector.tsx
      ModeSelector.tsx
      UserDropdown.tsx
    Sidebar/
      Sidebar.tsx
      QueryLogItem.tsx
    IntelligencePanel/
      IntelligencePanel.tsx
      ResponseSection.tsx
    EvidencePanel/
      EvidencePanel.tsx
      EvidenceItem.tsx
    QueryInputBar/
      QueryInputBar.tsx
```

## Tech

- React 18 (functional components, hooks)
- TypeScript
- Tailwind CSS
- Vite

Mock data: `src/data/mockData.ts`. Replace with API calls for production.
