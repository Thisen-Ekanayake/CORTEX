import { useState } from 'react'
import { WorkspaceSelector } from './WorkspaceSelector'
import { ModeSelector } from './ModeSelector'
import { UserDropdown } from './UserDropdown'
import type { IntelligenceMode, ModelId } from '../../types'

const models: { id: ModelId; label: string }[] = [
  { id: 'cortex-fast', label: 'Cortex Fast' },
  { id: 'cortex-pro', label: 'Cortex Pro' },
  { id: 'cortex-deep', label: 'Cortex Deep Analysis' },
]

type TopBarProps = {
  workspaceId: string
  onWorkspaceChange: (id: string) => void
  mode: IntelligenceMode
  onModeChange: (mode: IntelligenceMode) => void
  modelId: ModelId
  onModelChange: (id: ModelId) => void
}

export function TopBar({
  workspaceId,
  onWorkspaceChange,
  mode,
  onModeChange,
  modelId,
  onModelChange,
}: TopBarProps) {
  const [modelOpen, setModelOpen] = useState(false)
  const selectedModel = models.find((m) => m.id === modelId) ?? models[0]

  return (
    <header className="h-14 flex-shrink-0 flex items-center justify-between px-8 bg-surface-panel">
      <div className="flex items-center gap-8">
        <div className="text-sm font-semibold text-text-primary tracking-tight">
          Cortex
        </div>
        <WorkspaceSelector selectedId={workspaceId} onSelect={onWorkspaceChange} />
        <ModeSelector selected={mode} onSelect={onModeChange} />
        <div className="relative">
          <button
            type="button"
            onClick={() => setModelOpen(!modelOpen)}
            className="flex items-center gap-2 px-4 py-2.5 text-sm text-text-secondary bg-surface-hover rounded-xl hover:bg-surface-hover hover:text-text-primary transition-colors duration-200"
            aria-haspopup="listbox"
            aria-expanded={modelOpen}
          >
            <span>{selectedModel.label}</span>
            <span className="text-text-tertiary">&#9662;</span>
          </button>
          {modelOpen && (
            <ul
              role="listbox"
              className="absolute left-0 top-full mt-2 min-w-[200px] bg-surface-panel rounded-2xl shadow-panel z-50 py-2"
            >
              {models.map((m) => (
                <li key={m.id} role="option" aria-selected={modelId === m.id}>
                  <button
                    type="button"
                    onClick={() => {
                      onModelChange(m.id)
                      setModelOpen(false)
                    }}
                    className={`w-full text-left px-4 py-2.5 text-sm rounded-lg mx-1 transition-colors duration-200 ${
                      modelId === m.id
                        ? 'text-blue-400 bg-surface-hover'
                        : 'text-text-primary hover:bg-surface-hover'
                    }`}
                  >
                    {m.label}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-4 text-xs text-text-secondary">
          <span className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-500" />
            Local Processing
          </span>
          <span className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-500" />
            Index Synced
          </span>
          <span className="text-text-tertiary">Latency: — ms</span>
        </div>
        <UserDropdown />
      </div>
    </header>
  )
}
