import { useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useConversations } from '../../contexts/ConversationsContext'
import { useAuth } from '../../contexts/AuthContext'
import { createProject, type ConversationSummary } from '../../lib/backendApi'

type SidebarProps = {
  collapsed: boolean
  onToggle: () => void
}

const MAX_RECENTS = 10

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const { projects, recents, conversationsForProject, loading, error, refresh } = useConversations()
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  const handleNewProject = async () => {
    const name = window.prompt('Project name')
    if (!name || !name.trim()) return
    const project = await createProject(name.trim())
    await refresh()
    setExpanded((prev) => new Set(prev).add(project.id))
    navigate(`/project/${project.id}`)
  }

  const toggleProject = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })

  const navItemClass = ({ isActive }: { isActive: boolean }) =>
    `flex items-center gap-3 px-3 py-2 rounded-md transition-all duration-200 group relative overflow-hidden ${isActive
      ? 'text-accent-primary bg-accent-primary/10 shadow-glow'
      : 'text-text-secondary hover:text-text-primary hover:bg-white/5'
    }`

  const chatLinkClass = ({ isActive }: { isActive: boolean }) =>
    `flex items-center gap-3 px-3 py-2 rounded-md transition-all duration-200 text-sm truncate ${isActive
      ? 'text-accent-primary bg-accent-primary/10'
      : 'text-text-secondary hover:text-text-primary hover:bg-white/5'
    }`

  const handleLogout = () => {
    logout()
    navigate('/login', { replace: true })
  }

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
            <NavLink to="/" className={navItemClass} end>
              <span className="text-lg w-6 flex justify-center shrink-0">◉</span>
              <AnimatePresence>
                {!collapsed && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.1 }}
                    className="whitespace-nowrap overflow-hidden text-sm font-medium"
                  >
                    Overview
                  </motion.span>
                )}
              </AnimatePresence>
            </NavLink>
          </div>

          {!collapsed && (
            <>
              {loading && (
                <p className="px-3 text-xs text-text-tertiary">Loading…</p>
              )}
              {error && (
                <p className="px-3 text-xs text-red-400">{error}</p>
              )}
            </>
          )}

          {/* Projects Section */}
          {!collapsed && (
            <div>
              <div className="px-3 mb-2 flex items-center justify-between text-xs font-bold text-text-tertiary uppercase tracking-wider">
                <span>Projects</span>
                <button
                  onClick={handleNewProject}
                  title="New project"
                  aria-label="New project"
                  className="text-sm leading-none text-text-tertiary hover:text-text-primary transition-colors"
                >
                  ＋
                </button>
              </div>
              {projects.length === 0 && (
                <p className="px-3 text-xs text-text-tertiary normal-case font-normal tracking-normal">
                  No projects yet.
                </p>
              )}
              <div className="space-y-1">
                {projects.map((project) => {
                  const isOpen = expanded.has(project.id)
                  const projectChats = conversationsForProject(project.id)
                  return (
                    <div key={project.id}>
                      <div className="flex items-center">
                        <button
                          onClick={() => toggleProject(project.id)}
                          className="p-1 text-text-tertiary hover:text-text-primary transition-colors shrink-0"
                          aria-label={isOpen ? 'Collapse project' : 'Expand project'}
                        >
                          <span className="text-[10px] inline-block w-3">{isOpen ? '▾' : '▸'}</span>
                        </button>
                        <NavLink
                          to={`/project/${project.id}`}
                          className={({ isActive }) =>
                            `flex items-center gap-2 flex-1 px-2 py-2 rounded-md transition-all duration-200 text-sm truncate ${isActive
                              ? 'text-accent-primary bg-accent-primary/10'
                              : 'text-text-secondary hover:text-text-primary hover:bg-white/5'
                            }`
                          }
                        >
                          <span className="text-base shrink-0 opacity-70">{project.icon}</span>
                          <span className="truncate">{project.name}</span>
                        </NavLink>
                      </div>
                      {isOpen && (
                        <div className="ml-6 space-y-0.5 border-l border-border-subtle pl-2 mt-0.5">
                          {projectChats.length === 0 ? (
                            <p className="px-3 py-1 text-xs text-text-tertiary">No chats yet</p>
                          ) : (
                            projectChats.map((c) => (
                              <NavLink key={c.id} to={`/chat/${c.id}`} className={chatLinkClass}>
                                <span className="text-xs opacity-60 shrink-0">○</span>
                                <span className="truncate">{c.title}</span>
                              </NavLink>
                            ))
                          )}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Recents Section */}
          {!collapsed && (
            <div>
              <div className="px-3 mb-2 text-xs font-bold text-text-tertiary uppercase tracking-wider">
                Recent Chats
              </div>
              <div className="space-y-1">
                {!loading && recents.length === 0 ? (
                  <p className="px-3 text-xs text-text-tertiary">
                    No chats yet — start a new one.
                  </p>
                ) : (
                  recents.slice(0, MAX_RECENTS).map((c: ConversationSummary) => (
                    <NavLink key={c.id} to={`/chat/${c.id}`} className={chatLinkClass}>
                      <span className="text-base w-6 flex justify-center shrink-0 opacity-70">○</span>
                      <span className="truncate">{c.title}</span>
                    </NavLink>
                  ))
                )}
              </div>
            </div>
          )}
        </nav>
      </div>

      {/* Footer / User */}
      <div className="p-3 border-t border-border-subtle shrink-0 glass-panel mt-auto mx-3 mb-3 rounded-xl">
        <div className="flex items-center gap-2">
          <NavLink
            to="/settings"
            className="flex items-center gap-3 px-2 py-1.5 rounded-lg text-text-secondary hover:text-text-primary transition-colors flex-1 min-w-0"
          >
            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-accent-primary to-purple-500 shrink-0 flex items-center justify-center text-white text-xs font-bold">
              {user?.avatar_initials || (user?.display_name?.[0] ?? 'U').toUpperCase()}
            </div>
            <AnimatePresence>
              {!collapsed && (
                <motion.div
                  initial={{ opacity: 0, width: 0 }}
                  animate={{ opacity: 1, width: 'auto' }}
                  exit={{ opacity: 0, width: 0 }}
                  className="flex flex-col text-sm overflow-hidden whitespace-nowrap min-w-0"
                >
                  <span className="font-medium text-text-primary truncate">
                    {user?.display_name ?? 'Account'}
                  </span>
                  <span className="text-xs text-text-tertiary truncate">{user?.email}</span>
                </motion.div>
              )}
            </AnimatePresence>
          </NavLink>
          {!collapsed && (
            <button
              onClick={handleLogout}
              title="Sign out"
              aria-label="Sign out"
              className="p-2 rounded-lg text-text-tertiary hover:text-text-primary hover:bg-white/5 transition-colors shrink-0"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                <polyline points="16 17 21 12 16 7" />
                <line x1="21" y1="12" x2="9" y2="12" />
              </svg>
            </button>
          )}
        </div>
      </div>
    </motion.aside>
  )
}
