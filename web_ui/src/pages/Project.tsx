import { useParams, useNavigate, Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useConversations } from '../contexts/ConversationsContext'
import {
  renameProject,
  deleteProject,
  deleteConversation,
  updateConversation,
} from '../lib/backendApi'

export function Project() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { projects, conversationsForProject, loading, refresh } = useConversations()

  const project = projects.find((p) => p.id === id)
  const chats = id ? conversationsForProject(id) : []

  if (loading && !project) {
    return (
      <div className="p-8 text-text-tertiary text-sm">Loading project…</div>
    )
  }

  if (!project) {
    return (
      <div className="p-8">
        <p className="text-text-secondary">Project not found.</p>
        <Link to="/" className="text-accent-primary hover:underline text-sm">
          ← Back to Overview
        </Link>
      </div>
    )
  }

  const handleRenameProject = async () => {
    const name = window.prompt('Rename project', project.name)
    if (!name || name.trim() === project.name) return
    await renameProject(project.id, { name: name.trim() })
    refresh()
  }

  const handleDeleteProject = async () => {
    if (!window.confirm(`Delete project "${project.name}"? Its chats will be kept but unfiled.`)) {
      return
    }
    await deleteProject(project.id)
    refresh()
    navigate('/', { replace: true })
  }

  const handleRenameChat = async (chatId: string, current: string) => {
    const title = window.prompt('Rename chat', current)
    if (!title || title.trim() === current) return
    await updateConversation(chatId, { title: title.trim() })
    refresh()
  }

  const handleRemoveFromProject = async (chatId: string) => {
    await updateConversation(chatId, { project_id: null })
    refresh()
  }

  const handleDeleteChat = async (chatId: string) => {
    if (!window.confirm('Delete this chat permanently?')) return
    await deleteConversation(chatId)
    refresh()
  }

  return (
    <div className="p-4 md:p-8 max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-start justify-between mb-8"
      >
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-12 h-12 rounded-xl bg-accent-primary/10 border border-accent-primary/20 flex items-center justify-center text-2xl text-accent-primary shrink-0">
            {project.icon}
          </div>
          <div className="min-w-0">
            <h1 className="text-2xl font-semibold tracking-tight truncate">{project.name}</h1>
            <p className="text-sm text-text-tertiary">
              {chats.length} chat{chats.length === 1 ? '' : 's'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={handleRenameProject}
            className="text-sm px-3 py-2 rounded-lg text-text-secondary hover:text-text-primary hover:bg-white/5 transition-colors"
          >
            Rename
          </button>
          <button
            onClick={handleDeleteProject}
            className="text-sm px-3 py-2 rounded-lg text-red-400 hover:bg-red-500/10 transition-colors"
          >
            Delete
          </button>
        </div>
      </motion.div>

      <Link
        to={`/chat/new?project=${project.id}`}
        className="flex items-center gap-3 px-4 py-3 rounded-xl bg-surface-glass hover:bg-surface-glass-highlight border border-transparent hover:border-border-highlight text-text-primary transition-all mb-6"
      >
        <span className="text-xl">＋</span>
        <span className="font-medium">New chat in this project</span>
      </Link>

      <div className="space-y-2">
        {chats.length === 0 ? (
          <p className="text-text-tertiary text-sm px-1">
            No chats in this project yet.
          </p>
        ) : (
          chats.map((c) => (
            <div
              key={c.id}
              className="group flex items-center gap-3 glass-panel rounded-xl px-4 py-3 hover:bg-white/[0.04] transition-colors"
            >
              <Link to={`/chat/${c.id}`} className="flex items-center gap-3 flex-1 min-w-0">
                <span className="text-text-tertiary opacity-60">○</span>
                <span className="truncate text-text-primary">{c.title}</span>
              </Link>
              <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={() => handleRenameChat(c.id, c.title)}
                  className="text-xs px-2 py-1 rounded-md text-text-tertiary hover:text-text-primary hover:bg-white/5"
                >
                  Rename
                </button>
                <button
                  onClick={() => handleRemoveFromProject(c.id)}
                  className="text-xs px-2 py-1 rounded-md text-text-tertiary hover:text-text-primary hover:bg-white/5"
                  title="Move out of this project"
                >
                  Unfile
                </button>
                <button
                  onClick={() => handleDeleteChat(c.id)}
                  className="text-xs px-2 py-1 rounded-md text-red-400 hover:bg-red-500/10"
                >
                  Delete
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
