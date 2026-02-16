import type { CurrentQuery } from '../../types'
import { workspaces } from '../../data/mockData'
import { ResponseSection } from './ResponseSection'

type IntelligencePanelProps = {
  query: CurrentQuery | null
}

function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString([], {
    dateStyle: 'short',
    timeStyle: 'short',
  })
}

function workspaceLabel(id: string): string {
  return workspaces.find((w) => w.id === id)?.label ?? id
}

export function IntelligencePanel({ query }: IntelligencePanelProps) {
  if (!query) {
    return (
      <main className="flex-1 flex flex-col min-w-0 bg-surface overflow-auto">
        <div className="flex-1 flex items-center justify-center text-text-tertiary text-sm px-8">
          Select a query from the log or enter a new query below.
        </div>
      </main>
    )
  }

  return (
    <main className="flex-1 flex flex-col min-w-0 bg-surface overflow-auto">
      <div className="flex-1 px-8 py-8 space-y-8 max-w-4xl">
        <div className="bg-surface-panel rounded-2xl shadow-panel p-6">
          <h3 className="text-xs font-medium text-text-tertiary uppercase tracking-wider mb-3">
            Query
          </h3>
          <p className="text-base text-text-primary font-medium mb-3 leading-relaxed">
            {query.queryText}
          </p>
          <div className="flex flex-wrap gap-4 text-sm text-text-tertiary">
            <span>Workspace: {workspaceLabel(query.workspace)}</span>
            <span>Mode: {query.mode}</span>
            <span>{formatTimestamp(query.timestamp)}</span>
          </div>
        </div>
        <div>
          <h3 className="text-xs font-medium text-text-tertiary uppercase tracking-wider mb-4">
            Response
          </h3>
          <ResponseSection response={query.response} />
        </div>
      </div>
    </main>
  )
}
