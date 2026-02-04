import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card } from '../components/ui/Card'
import { documents } from '../data/placeholder'
import './pages.css'

type ViewMode = 'grid' | 'list'

export function Documents() {
  const [view, setView] = useState<ViewMode>('grid')
  const [hoverId, setHoverId] = useState<string | null>(null)
  const [filter, setFilter] = useState('')

  const filtered = documents.filter(
    (d) =>
      d.name.toLowerCase().includes(filter.toLowerCase()) ||
      d.type.toLowerCase().includes(filter.toLowerCase())
  )

  return (
    <motion.div
      className="page page--documents"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <h2 className="page__title">Documents</h2>
      <p className="page__subtitle">
        Interactive list with hover previews. Filter by name or type.
      </p>

      <div className="doc-toolbar">
        <input
          type="search"
          className="doc-toolbar__filter"
          placeholder="Filter documents…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label="Filter documents"
        />
        <div className="doc-toolbar__view">
          <button
            type="button"
            className={view === 'grid' ? 'active' : ''}
            onClick={() => setView('grid')}
            aria-label="Grid view"
          >
            ▦
          </button>
          <button
            type="button"
            className={view === 'list' ? 'active' : ''}
            onClick={() => setView('list')}
            aria-label="List view"
          >
            ≡
          </button>
        </div>
      </div>

      <motion.div
        className={`doc-browser doc-browser--${view}`}
        layout
        transition={{ duration: 0.25 }}
      >
        <AnimatePresence mode="popLayout">
          {filtered.map((d, i) => (
            <motion.div
              key={d.id}
              layout
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ delay: i * 0.03 }}
              onMouseEnter={() => setHoverId(d.id)}
              onMouseLeave={() => setHoverId(null)}
            >
              <Card className={`doc-card ${hoverId === d.id ? 'doc-card--hover' : ''}`}>
                <div className="doc-card__header">
                  <span className="doc-card__icon">{d.type === 'Markdown' ? '◉' : '▤'}</span>
                  <span className="doc-card__name">{d.name}</span>
                  <span className="doc-card__meta">{d.type} · {d.size}</span>
                </div>
                <AnimatePresence>
                  {hoverId === d.id && (
                    <motion.div
                      className="doc-card__preview"
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      {d.preview}
                    </motion.div>
                  )}
                </AnimatePresence>
                <span className="doc-card__updated">{d.updated}</span>
              </Card>
            </motion.div>
          ))}
        </AnimatePresence>
      </motion.div>

      {filtered.length === 0 && (
        <p className="page__empty">No documents match “{filter}”.</p>
      )}
    </motion.div>
  )
}
