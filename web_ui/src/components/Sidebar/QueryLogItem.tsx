import type { QueryLogEntry } from '../../types'
import { workspaces } from '../../data/mockData'

type QueryLogItemProps = {
  entry: QueryLogEntry
  isActive?: boolean
  onSelect?: () => void
}

function formatTime(iso: string): string {
  const d = new Date(iso)
  const now = new Date()
  const diffMs = now.getTime() - d.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  if (diffMins < 60) return `${diffMins}m ago`
  const diffHours = Math.floor(diffMins / 60)
  if (diffHours < 24) return `${diffHours}h ago`
  return d.toLocaleDateString()
}

function workspaceLabel(id: string): string {
  return workspaces.find((w) => w.id === id)?.label ?? id
}

export function QueryLogItem({ entry, isActive, onSelect }: QueryLogItemProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full text-left px-4 py-3 rounded-xl transition-colors duration-200 ${
        isActive
          ? 'bg-surface-hover text-text-primary ring-1 ring-blue-500/30'
          : 'text-text-secondary hover:bg-surface-hover hover:text-text-primary'
      }`}
    >
      <div className="text-sm font-medium text-text-primary truncate" title={entry.title}>
        {entry.title}
      </div>
      <div className="flex items-center gap-2 mt-1.5 text-xs text-text-tertiary">
        <span>{workspaceLabel(entry.workspace)}</span>
        <span>{entry.mode}</span>
        <span>{formatTime(entry.timestamp)}</span>
      </div>
      <div className="mt-1 text-xs text-text-tertiary">
        Confidence: {(entry.confidence * 100).toFixed(0)}%
      </div>
    </button>
  )
}
