import { NavLink } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'

const navItems = [
  { path: '/', icon: '◉', label: 'Overview' },
  { path: '/search', icon: '⌕', label: 'Search' },
  { path: '/documents', icon: '▤', label: 'Documents' },
  { path: '/chat', icon: '◈', label: 'Assist' },
  { path: '/settings', icon: '⚙', label: 'Settings' },
] as const

type SidebarProps = {
  collapsed: boolean
  onToggle: () => void
}

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  return (
    <motion.aside
      className="sidebar"
      initial={false}
      animate={{ width: collapsed ? 'var(--sidebar-collapsed)' : 'var(--sidebar-width)' }}
      transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
      aria-label="Main navigation"
    >
      <div className="sidebar__inner">
        <button
          type="button"
          className="sidebar__toggle"
          onClick={onToggle}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-expanded={!collapsed}
        >
          <span className="sidebar__toggle-icon">{collapsed ? '›' : '‹'}</span>
        </button>
        <nav className="sidebar__nav">
          {navItems.map(({ path, icon, label }) => (
            <NavLink
              key={path}
              to={path}
              className={({ isActive }) =>
                `sidebar__link ${isActive ? 'sidebar__link--active' : ''}`
              }
              end={path === '/'}
              aria-current={path === window.location.pathname ? 'page' : undefined}
            >
              <span className="sidebar__icon" aria-hidden>{icon}</span>
              <AnimatePresence mode="wait">
                {!collapsed && (
                  <motion.span
                    className="sidebar__label"
                    initial={{ opacity: 0, width: 0 }}
                    animate={{ opacity: 1, width: 'auto' }}
                    exit={{ opacity: 0, width: 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    {label}
                  </motion.span>
                )}
              </AnimatePresence>
            </NavLink>
          ))}
        </nav>
      </div>
    </motion.aside>
  )
}
