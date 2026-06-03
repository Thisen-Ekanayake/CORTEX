import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card } from '../components/ui/Card'
import { listDocuments, uploadDocument, type DocumentInfo } from '../lib/api'
import './pages.css'

type ViewMode = 'grid' | 'list'

export function Documents() {
  const [view, setView] = useState<ViewMode>('grid')
  const [hoverId, setHoverId] = useState<string | null>(null)
  const [filter, setFilter] = useState('')
  const [docs, setDocs] = useState<DocumentInfo[]>([])
  const [indexedCount, setIndexedCount] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await listDocuments()
      setDocs(res.documents)
      setIndexedCount(res.indexedCount)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load documents')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    setNotice(null)
    setError(null)
    try {
      const res = await uploadDocument(file)
      setNotice(`Ingested ${res.name} — ${res.chunksAdded} chunk${res.chunksAdded === 1 ? '' : 's'} added.`)
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const filtered = docs.filter(
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
        {indexedCount} document{indexedCount === 1 ? '' : 's'} in the knowledge base. Upload to ingest into the vector store.
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
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt,.docx"
          onChange={handleUpload}
          style={{ display: 'none' }}
        />
        <button
          type="button"
          className="doc-toolbar__upload"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? 'Ingesting…' : '⤓ Upload'}
        </button>
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

      {notice && <p className="page__notice">{notice}</p>}
      {error && <p className="page__error">{error}</p>}
      {loading && <p className="page__subtitle">Loading…</p>}

      <motion.div
        className={`doc-browser doc-browser--${view}`}
        layout
        transition={{ duration: 0.25 }}
      >
        <AnimatePresence mode="popLayout">
          {filtered.map((d, i) => (
            <motion.div
              key={d.name}
              layout
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ delay: i * 0.03 }}
              onMouseEnter={() => setHoverId(d.name)}
              onMouseLeave={() => setHoverId(null)}
            >
              <Card className={`doc-card ${hoverId === d.name ? 'doc-card--hover' : ''}`}>
                <div className="doc-card__header">
                  <span className="doc-card__icon">{d.type === 'Markdown' ? '◉' : '▤'}</span>
                  <span className="doc-card__name">{d.name}</span>
                  <span className="doc-card__meta">{d.type} · {d.size}</span>
                </div>
                <span className="doc-card__updated">{d.updated}</span>
              </Card>
            </motion.div>
          ))}
        </AnimatePresence>
      </motion.div>

      {!loading && filtered.length === 0 && (
        <p className="page__empty">
          {docs.length === 0 ? 'No documents yet. Upload one to get started.' : `No documents match “${filter}”.`}
        </p>
      )}
    </motion.div>
  )
}
