import {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  useMemo,
  type ReactNode,
} from 'react'
import {
  listProjects,
  listConversations,
  type Project,
  type ConversationSummary,
} from '../lib/backendApi'

interface ConversationsContextValue {
  projects: Project[]
  conversations: ConversationSummary[]
  /** Conversations not filed under any project, most-recent first. */
  recents: ConversationSummary[]
  conversationsForProject: (projectId: string) => ConversationSummary[]
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
}

const ConversationsContext = createContext<ConversationsContextValue | undefined>(undefined)

export function ConversationsProvider({ children }: { children: ReactNode }) {
  const [projects, setProjects] = useState<Project[]>([])
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setError(null)
    try {
      const [proj, convs] = await Promise.all([listProjects(), listConversations()])
      setProjects(proj)
      setConversations(convs)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load chat history')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const recents = useMemo(
    () => conversations.filter((c) => !c.project_id),
    [conversations],
  )

  const conversationsForProject = useCallback(
    (projectId: string) => conversations.filter((c) => c.project_id === projectId),
    [conversations],
  )

  return (
    <ConversationsContext.Provider
      value={{
        projects,
        conversations,
        recents,
        conversationsForProject,
        loading,
        error,
        refresh,
      }}
    >
      {children}
    </ConversationsContext.Provider>
  )
}

export function useConversations(): ConversationsContextValue {
  const ctx = useContext(ConversationsContext)
  if (!ctx) {
    throw new Error('useConversations must be used within a ConversationsProvider')
  }
  return ctx
}
