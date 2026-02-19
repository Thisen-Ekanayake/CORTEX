import { useState } from 'react'
import { QueryLogItem } from './QueryLogItem'
import { queryLog, dataSourceStatus } from '../../data/mockData'
type SidebarProps = {
  selectedQueryId: string | null
  onSelectQuery: () => void
}

function formatSyncTime(iso: string): string {
  const d = new Date(iso)
  const now = new Date()
  const diffMs = now.getTime() - d.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function statusBadge(status: string) {
  switch (status) {
    case 'synced':
      return (
        <span className="inline-flex items-center px-2 py-1 text-xs font-medium rounded-lg bg-green-500/15 text-green-400">
          Synced
        </span>
      )
    case 'syncing':
      return (
        <span className="inline-flex items-center px-2 py-1 text-xs font-medium rounded-lg bg-amber-500/15 text-amber-400">
          Syncing
        </span>
      )
    default:
      return (
        <span className="inline-flex items-center px-2 py-1 text-xs font-medium rounded-lg bg-surface-hover text-text-tertiary">
          Pending
        </span>
      )
  }
}

export function Sidebar({ selectedQueryId, onSelectQuery }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false)

  if (collapsed) {
    return (
      <aside className="w-14 flex-shrink-0 flex flex-col bg-surface">
        <button
          type="button"
          onClick={() => setCollapsed(false)}
          className="p-3 text-text-tertiary hover:text-text-primary transition-colors duration-200"
          aria-label="Expand sidebar"
        >
          &#9654;
        </button>
      </aside>
    )
  }

  return (
    <aside className="w-72 flex-shrink-0 flex flex-col bg-surface py-5 pl-5 pr-3">
      <div className="flex items-center justify-between mb-6">
        <span className="text-xs font-medium text-text-tertiary uppercase tracking-wider">
          Knowledge & Navigation
        </span>
        <button
          type="button"
          onClick={() => setCollapsed(true)}
          className="p-2 text-text-tertiary hover:text-text-primary rounded-lg hover:bg-surface-hover transition-colors duration-200"
          aria-label="Collapse sidebar"
        >
          &#9664;
        </button>
      </div>
      <div className="flex-1 overflow-y-auto space-y-8">
        <section>
          <h3 className="text-xs font-medium text-text-tertiary uppercase tracking-wider mb-3 px-1">
            Workspaces
          </h3>
          <ul className="space-y-1">
            <li>
              <button
                type="button"
                className="w-full text-left px-4 py-2.5 text-sm text-text-primary hover:bg-surface-hover rounded-xl transition-colors duration-200"
              >
                HR
              </button>
            </li>
            <li>
              <button
                type="button"
                className="w-full text-left px-4 py-2.5 text-sm text-text-primary hover:bg-surface-hover rounded-xl transition-colors duration-200"
              >
                Finance
              </button>
            </li>
            <li>
              <button
                type="button"
                className="w-full text-left px-4 py-2.5 text-sm text-text-primary hover:bg-surface-hover rounded-xl transition-colors duration-200"
              >
                Legal
              </button>
            </li>
            <li>
              <button
                type="button"
                className="w-full text-left px-4 py-2.5 text-sm text-text-primary hover:bg-surface-hover rounded-xl transition-colors duration-200"
              >
                R&D
              </button>
            </li>
          </ul>
        </section>
        <section>
          <h3 className="text-xs font-medium text-text-tertiary uppercase tracking-wider mb-3 px-1">
            Query Log
          </h3>
          <ul className="space-y-1">
            {queryLog.map((entry) => (
              <li key={entry.id}>
                <QueryLogItem
                  entry={entry}
                  isActive={selectedQueryId === entry.id}
                  onSelect={onSelectQuery}
                />
              </li>
            ))}
          </ul>
        </section>
        <section>
          <h3 className="text-xs font-medium text-text-tertiary uppercase tracking-wider mb-3 px-1">
            Data Sources
          </h3>
          <div className="space-y-2.5 text-sm text-text-secondary px-1">
            <div>Indexed: {dataSourceStatus.indexedCount.toLocaleString()} documents</div>
            <div>Last sync: {formatSyncTime(dataSourceStatus.lastSyncTime)}</div>
            <div>{statusBadge(dataSourceStatus.status)}</div>
          </div>
        </section>
        <section>
          <h3 className="text-xs font-medium text-text-tertiary uppercase tracking-wider mb-3 px-1">
            Admin
          </h3>
          <div className="text-sm text-text-tertiary px-1">Placeholder</div>
        </section>
      </div>
    </aside>
  )
}
