import { useEffect, useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { listDocuments, type DocumentInfo } from '../../lib/api'

type SortField = 'name' | 'size' | 'modified'
type SortDir = 'asc' | 'desc'

const SORT_FIELDS: { id: SortField; label: string }[] = [
  { id: 'name', label: 'Name' },
  { id: 'size', label: 'Size' },
  { id: 'modified', label: 'Modified' },
]

function compare(a: DocumentInfo, b: DocumentInfo, field: SortField): number {
  switch (field) {
    case 'name':
      return a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
    case 'size':
      return a.sizeBytes - b.sizeBytes
    case 'modified':
      return new Date(a.modifiedAt).getTime() - new Date(b.modifiedAt).getTime()
  }
}

export function ContextPanel() {
  const [docs, setDocs] = useState<DocumentInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [sortField, setSortField] = useState<SortField>('name')
  const [sortDir, setSortDir] = useState<SortDir>('asc')
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    let cancelled = false
    listDocuments()
      .then((res) => {
        if (!cancelled) setDocs(res.documents)
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load documents')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const sorted = useMemo(() => {
    const out = [...docs].sort((a, b) => compare(a, b, sortField))
    return sortDir === 'asc' ? out : out.reverse()
  }, [docs, sortField, sortDir])

  function changeSort(field: SortField) {
    if (field === sortField) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortField(field)
      setSortDir('asc')
    }
  }

  // Collapsed rail: a thin strip showing vertical "Sources" text; click to expand.
  if (collapsed) {
    return (
      <aside className="w-10 border-l border-border-subtle bg-surface-glass/30 hidden lg:flex flex-col items-center py-3 shrink-0">
        <button
          type="button"
          onClick={() => setCollapsed(false)}
          aria-label="Show sources panel"
          aria-expanded={false}
          title="Show sources"
          className="flex flex-col items-center gap-2 text-text-secondary hover:text-text-primary transition-colors"
        >
          <span className="text-lg leading-none">‹</span>
          <span
            style={{ writingMode: 'vertical-rl' }}
            className="text-xs font-medium uppercase tracking-wider"
          >
            Sources
          </span>
        </button>
      </aside>
    )
  }

  return (
    <aside className="w-80 border-l border-border-subtle bg-surface-glass/30 flex flex-col hidden lg:flex shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-subtle">
        <div className="text-sm font-medium pb-2 border-b-2 border-accent-primary text-text-primary">
          Sources
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-text-tertiary">
            {loading ? '—' : `${docs.length} indexed`}
          </span>
          <button
            type="button"
            onClick={() => setCollapsed(true)}
            aria-label="Hide sources panel"
            aria-expanded={true}
            title="Hide sources"
            className="p-1 -mr-1 text-text-tertiary hover:text-text-primary rounded-md hover:bg-white/5 transition-colors"
          >
            <span className="text-lg leading-none">›</span>
          </button>
        </div>
      </div>

      {/* Sort controls */}
      <div className="flex items-center gap-1 px-4 py-2 border-b border-border-subtle">
        <span className="text-xs text-text-tertiary mr-1">Sort</span>
        {SORT_FIELDS.map((f) => {
          const active = sortField === f.id
          return (
            <button
              key={f.id}
              type="button"
              onClick={() => changeSort(f.id)}
              aria-pressed={active}
              title={`Sort by ${f.label}${active ? (sortDir === 'asc' ? ' (ascending)' : ' (descending)') : ''}`}
              className={`flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium transition-colors ${
                active
                  ? 'bg-accent-primary/15 text-accent-primary'
                  : 'text-text-secondary hover:text-text-primary hover:bg-white/5'
              }`}
            >
              {f.label}
              <span className="text-[10px] leading-none w-2 inline-block">
                {active ? (sortDir === 'asc' ? '↑' : '↓') : ''}
              </span>
            </button>
          )
        })}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
        {error && <p className="text-xs text-accent-error">{error}</p>}

        {loading && !error && (
          <div className="space-y-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-16 rounded-lg bg-white/5 animate-pulse" />
            ))}
          </div>
        )}

        {!loading && !error && sorted.length === 0 && (
          <p className="text-xs text-text-tertiary">
            No documents indexed yet. Upload one on the Documents page.
          </p>
        )}

        {!loading && !error && sorted.length > 0 && (
          <motion.div layout className="space-y-3">
            <AnimatePresence mode="popLayout">
              {sorted.map((doc) => (
                <motion.div
                  key={doc.name}
                  layout
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ duration: 0.18 }}
                  className="p-3 rounded-lg border border-border-subtle bg-bg-elevated/50"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-accent-success text-xs">●</span>
                    <span className="text-xs font-medium text-text-secondary uppercase">
                      {doc.type}
                    </span>
                  </div>
                  <h4 className="text-sm font-medium text-text-primary mb-1 break-words">
                    {doc.name}
                  </h4>
                  <div className="flex items-center justify-between text-xs text-text-tertiary">
                    <span>{doc.size}</span>
                    <span>{doc.updated}</span>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </motion.div>
        )}
      </div>
    </aside>
  )
}
