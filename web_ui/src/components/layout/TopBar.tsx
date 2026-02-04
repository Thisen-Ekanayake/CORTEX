import { useState, useRef, useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { VoiceCommand } from '../voice/VoiceCommand'

const pageTitles: Record<string, string> = {
  '/': 'Overview',
  '/search': 'Search',
  '/documents': 'Documents',
  '/chat': 'Assist',
  '/settings': 'Settings',
}

export function TopBar() {
  const location = useLocation()
  const navigate = useNavigate()
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [notificationsOpen, setNotificationsOpen] = useState(false)
  const [contextOpen, setContextOpen] = useState(false)
  const searchInputRef = useRef<HTMLInputElement>(null)

  const title = pageTitles[location.pathname] ?? 'CORTEX'

  useEffect(() => {
    if (searchOpen) searchInputRef.current?.focus()
  }, [searchOpen])

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (searchQuery.trim()) {
      navigate(`/search?q=${encodeURIComponent(searchQuery.trim())}`)
      setSearchOpen(false)
      setSearchQuery('')
    }
  }

  return (
    <header className="topbar" role="banner">
      <div className="topbar__left">
        <h1 className="topbar__title">{title}</h1>
      </div>
      <div className="topbar__center">
        <form
          className={`topbar__search ${searchOpen ? 'topbar__search--open' : ''}`}
          onSubmit={handleSearchSubmit}
          role="search"
        >
          <input
            ref={searchInputRef}
            type="search"
            placeholder="Search documents & answers…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onBlur={() => !searchQuery && setSearchOpen(false)}
            className="topbar__search-input"
            aria-label="Search"
            aria-expanded={searchOpen}
          />
          <button
            type="button"
            className="topbar__search-trigger"
            onClick={() => setSearchOpen(true)}
            aria-label="Open search"
          >
            ⌕
          </button>
        </form>
      </div>
      <div className="topbar__right">
        <VoiceCommand />
        <div className="topbar__actions">
          <button
            type="button"
            className="topbar__btn"
            onClick={() => setContextOpen((o) => !o)}
            aria-label="Context menu"
            aria-haspopup="menu"
            aria-expanded={contextOpen}
          >
            ⋮
          </button>
          <button
            type="button"
            className="topbar__btn topbar__btn--notify"
            onClick={() => setNotificationsOpen((o) => !o)}
            aria-label="Notifications (3)"
            aria-expanded={notificationsOpen}
          >
            ◉
            <span className="topbar__badge" aria-hidden>3</span>
          </button>
          <button type="button" className="topbar__avatar" aria-label="Profile">
            <span>C</span>
          </button>
        </div>
      </div>
      <AnimatePresence>
        {contextOpen && (
          <motion.div
            className="topbar__dropdown topbar__dropdown--context"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.15 }}
          >
            <button type="button">New query</button>
            <button type="button">Export</button>
            <button type="button">Help</button>
          </motion.div>
        )}
        {notificationsOpen && (
          <motion.div
            className="topbar__dropdown topbar__dropdown--notify"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.15 }}
          >
            <div className="topbar__notify-item">
              <strong>RAG index updated</strong>
              <span>3 new documents</span>
            </div>
            <div className="topbar__notify-item">
              <strong>Router accuracy</strong>
              <span>95% this week</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  )
}
