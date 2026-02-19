import type { EvidenceItem as EvidenceItemType } from '../../types'

type EvidenceItemProps = {
  item: EvidenceItemType
}

export function EvidenceItem({ item }: EvidenceItemProps) {
  return (
    <div className="bg-surface-panel rounded-2xl shadow-panel-sm p-4">
      <div className="flex items-center justify-between gap-3 mb-3">
        <span className="flex-shrink-0 w-8 h-8 flex items-center justify-center text-sm font-medium text-text-tertiary bg-surface-hover rounded-xl">
          {item.rank}
        </span>
        <span className="text-sm font-medium text-text-primary truncate min-w-0" title={item.documentName}>
          {item.documentName}
        </span>
      </div>
      <div className="flex items-center gap-3 text-xs text-text-tertiary mb-3">
        <span>Similarity: {(item.similarityScore * 100).toFixed(0)}%</span>
        {item.pageNumber != null && <span>Page {item.pageNumber}</span>}
      </div>
      <p className="text-sm text-text-secondary leading-relaxed line-clamp-3 mb-4">
        {item.snippet}
      </p>
      <button
        type="button"
        className="w-full px-4 py-2.5 text-sm text-blue-400 bg-blue-500/10 rounded-xl hover:bg-blue-500/20 transition-colors duration-200"
      >
        Open Document
      </button>
    </div>
  )
}
