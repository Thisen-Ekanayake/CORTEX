import { NavLink } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'

type SidebarProps = {
  collapsed: boolean
  onToggle: () => void
}

const SECTIONS = {
  main: [
    { path: '/', icon: '◉', label: 'Overview' },
  ],
  projects: [
    { path: '/project/alpha', icon: '◰', label: 'Project Alpha' },
    { path: '/project/beta', icon: '◳', label: 'Marketing Q4' },
  ],
  recents: [
    { path: '/chat/1', icon: '○', label: 'Competitor Analysis' },
    { path: '/chat/2', icon: '○', label: 'Q3 Financials' },
  ]
}

export function Sidebar({ collapsed, onToggle }: SidebarProps) {

  const navItemClass = ({ isActive }: { isActive: boolean }) =>
    `flex items-center gap-3 px-3 py-2 rounded-md transition-all duration-200 group relative overflow-hidden ${isActive
      ? 'text-accent-primary bg-accent-primary/10 shadow-glow'
      : 'text-text-secondary hover:text-text-primary hover:bg-white/5'
    }`

  return (
    <motion.aside
      className="h-screen flex flex-col glass-sidebar relative z-50"
      initial={false}
      animate={{
        width: collapsed ? 'var(--sidebar-collapsed)' : 'var(--sidebar-width)'
      }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
    >
      {/* Header / Toggle */}
      <div className="h-[var(--top-bar-height)] flex items-center px-4 border-b border-border-subtle shrink-0 justify-between">
        <div className="flex items-center gap-3 overflow-hidden">
          <div className="w-8 h-8 rounded-lg bg-accent-primary/20 flex items-center justify-center text-accent-primary shrink-0">
            C
          </div>
          <AnimatePresence>
            {!collapsed && (
              <motion.span
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.2 }}
                className="font-semibold text-text-primary tracking-tight whitespace-nowrap"
              >
                CORTEX
              </motion.span>
            )}
          </AnimatePresence>
        </div>

        <button
          onClick={onToggle}
          className="p-1.5 hover:bg-white/10 rounded-md text-text-secondary transition-colors absolute right-4 top-5"
          style={{ right: collapsed ? '50%' : '16px', transform: collapsed ? 'translateX(50%)' : 'none' }}
        >
          <span className="text-lg leading-none">{collapsed ? '›' : '‹'}</span>
        </button>
      </div>

      {/* Main Navigation */}
      <div className="flex-1 overflow-y-auto py-6 custom-scrollbar flex flex-col gap-6 px-3">
        {/* New Chat Button */}
        <NavLink
          to="/chat/new"
          className={({ isActive }) =>
            `flex items-center gap-3 px-3 py-3 rounded-xl transition-all duration-200 group border border-transparent ${isActive
              ? 'bg-accent-primary text-white shadow-glow border-accent-highlight'
              : 'bg-surface-glass hover:bg-surface-glass-highlight hover:border-border-highlight text-text-primary'
            }`
          }
        >
          <span className="text-xl flex items-center justify-center w-6 h-6">＋</span>
          <AnimatePresence>
            {!collapsed && (
              <motion.span
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: 'auto' }}
                exit={{ opacity: 0, width: 0 }}
                className="font-medium whitespace-nowrap overflow-hidden"
              >
                New Chat
              </motion.span>
            )}
          </AnimatePresence>
        </NavLink>

        <nav className="space-y-6">
          {/* Main Links */}
          <div className="space-y-1">
            {SECTIONS.main.map((item) => (
              <NavLink key={item.path} to={item.path} className={navItemClass}>
                <span className="text-lg w-6 flex justify-center shrink-0">{item.icon}</span>
                <AnimatePresence>
                  {!collapsed && (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.1 }}
                      className="whitespace-nowrap overflow-hidden text-sm font-medium"
                    >
                      {item.label}
                    </motion.span>
                  )}
                </AnimatePresence>
              </NavLink>
            ))}
          </div>

          {/* Projects Section */}
          <div>
            {!collapsed && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="px-3 mb-2 flex items-center justify-between text-xs font-bold text-text-tertiary uppercase tracking-wider"
              >
                <span>Projects</span>
              </motion.div>
            )}
            <div className="space-y-1">
              {SECTIONS.projects.map((item) => (
                <NavLink key={item.path} to={item.path} className={navItemClass}>
                  <span className="text-base w-6 flex justify-center shrink-0 opacity-70">{item.icon}</span>
                  <AnimatePresence>
                    {!collapsed && (
                      <motion.span
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.1 }}
                        className="whitespace-nowrap overflow-hidden text-sm"
                      >
                        {item.label}
                      </motion.span>
                    )}
                  </AnimatePresence>
                </NavLink>
              ))}
            </div>
          </div>

          {/* Recents Section */}
          <div>
            {!collapsed && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="px-3 mb-2 text-xs font-bold text-text-tertiary uppercase tracking-wider"
              >
                Recent Chats
              </motion.div>
            )}
            <div className="space-y-1">
              {SECTIONS.recents.map((item) => (
                <NavLink key={item.path} to={item.path} className={navItemClass}>
                  <span className="text-base w-6 flex justify-center shrink-0 opacity-70">{item.icon}</span>
                  <AnimatePresence>
                    {!collapsed && (
                      <motion.span
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.1 }}
                        className="whitespace-nowrap overflow-hidden text-sm"
                      >
                        {item.label}
                      </motion.span>
                    )}
                  </AnimatePresence>
                </NavLink>
              ))}
            </div>
          </div>
        </nav>
      </div>

      {/* Footer / User */}
      <div className="p-3 border-t border-border-subtle shrink-0 glass-panel mt-auto mx-3 mb-3 rounded-xl">
        <NavLink
          to="/settings"
          className="flex items-center gap-3 px-2 py-1.5 rounded-lg text-text-secondary hover:text-text-primary transition-colors"
        >
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-accent-primary to-purple-500 shrink-0" />
          <AnimatePresence>
            {!collapsed && (
              <motion.div
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: 'auto' }}
                exit={{ opacity: 0, width: 0 }}
                className="flex flex-col text-sm overflow-hidden whitespace-nowrap"
              >
                <span className="font-medium text-text-primary">Admin User</span>
                <span className="text-xs text-text-tertiary">Pro Plan</span>
              </motion.div>
            )}
          </AnimatePresence>
        </NavLink>
      </div>
    </motion.aside>
  )
}
