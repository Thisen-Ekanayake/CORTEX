import { useState } from 'react'
import { EvidenceItem } from './EvidenceItem'
import type { EvidenceItem as EvidenceItemType } from '../../types'

type EvidencePanelProps = {
  items: EvidenceItemType[]
}

export function EvidencePanel({ items }: EvidencePanelProps) {
  const [collapsed, setCollapsed] = useState(false)

  if (collapsed) {
    return (
      <aside className="w-12 flex-shrink-0 flex flex-col bg-surface">
        <button
          type="button"
          onClick={() => setCollapsed(false)}
          className="flex-1 flex items-center justify-center text-text-tertiary hover:text-text-primary transition-colors duration-200"
          aria-label="Expand evidence panel"
          title="Evidence"
        >
          <span className="text-xs uppercase tracking-wider" style={{ writingMode: 'vertical-rl' }}>
            Evidence
          </span>
        </button>
      </aside>
    )
  }

  return (
    <aside className="w-96 flex-shrink-0 flex flex-col bg-surface py-5 pr-5 pl-3">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xs font-medium text-text-tertiary uppercase tracking-wider">
          Evidence
        </h3>
        <button
          type="button"
          onClick={() => setCollapsed(true)}
          className="p-2 text-text-tertiary hover:text-text-primary rounded-lg hover:bg-surface-hover transition-colors duration-200"
          aria-label="Collapse evidence panel"
        >
          &#9654;
        </button>
      </div>
      <div className="flex-1 overflow-y-auto space-y-4 pr-2">
        {items.length === 0 ? (
          <p className="text-sm text-text-tertiary">No evidence retrieved.</p>
        ) : (
          items.map((item) => <EvidenceItem key={item.rank} item={item} />)
        )}
      </div>
    </aside>
  )
}
