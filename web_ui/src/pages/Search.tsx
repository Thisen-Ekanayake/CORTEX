import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Card } from '../components/ui/Card'
import { searchResults } from '../data/placeholder'
import './pages.css'

export function Search() {
  const [searchParams] = useSearchParams()
  const q = searchParams.get('q') ?? ''
  const [query, setQuery] = useState(q)
  const [results, setResults] = useState<typeof searchResults>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setQuery(q)
    if (q) {
      setLoading(true)
      const t = setTimeout(() => {
        setResults(searchResults.map((r) => ({ ...r, score: 0.8 + Math.random() * 0.2 })))
        setLoading(false)
      }, 600)
      return () => clearTimeout(t)
    } else {
      setResults([])
    }
  }, [q])

  return (
    <motion.div
      className="page page--search"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <h2 className="page__title">Search</h2>
      <p className="page__subtitle">
        Fast query with animated results. Try a question or keyword.
      </p>

      <div className="search-box">
        <input
          type="search"
          className="search-box__input"
          placeholder="Search documents & answers…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && (window.location.href = `/search?q=${encodeURIComponent(query)}`)}
          aria-label="Search query"
        />
        <button
          type="button"
          className="search-box__btn"
          onClick={() => (window.location.href = `/search?q=${encodeURIComponent(query)}`)}
          aria-label="Run search"
        >
          ⌕ Search
        </button>
      </div>

      {loading && (
        <motion.div
          className="search-loading"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <span className="search-loading__dot" />
          <span className="search-loading__dot" />
          <span className="search-loading__dot" />
        </motion.div>
      )}

      <AnimatePresence mode="wait">
        {!loading && results.length > 0 && (
          <motion.ul
            className="search-results"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            {results.map((r, i) => (
              <motion.li
                key={r.id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
              >
                <Card>
                  <div className="search-result">
                    <h4 className="search-result__title">{r.title}</h4>
                    <p className="search-result__snippet">{r.snippet}</p>
                    <div className="search-result__meta">
                      <span className="search-result__source">{r.source}</span>
                      <span className="search-result__score">{(r.score * 100).toFixed(0)}% match</span>
                    </div>
                  </div>
                </Card>
              </motion.li>
            ))}
          </motion.ul>
        )}
        {!loading && q && results.length === 0 && (
          <motion.p
            className="page__empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            No results for “{q}”. Try different keywords.
          </motion.p>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
