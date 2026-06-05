/**
 * Typed client for the user-management `backend/` service (auth, projects,
 * conversations). All calls go through the relative `/backend-api` path, which
 * Vite rewrites to `/api` on http://localhost:8000 (see vite.config.ts).
 *
 * The cortex_api AI service (chat/RAG/search/voice/vision) is a separate client
 * in ./api.ts and is unauthenticated.
 */

const BASE = '/backend-api'
const TOKEN_KEY = 'cortex.token'

// ── Token storage ─────────────────────────────────────────────────────────────

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

/** Thrown when the backend rejects credentials (401) so callers can sign out. */
export class UnauthorizedError extends Error {
  constructor(message = 'Session expired') {
    super(message)
    this.name = 'UnauthorizedError'
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken()
  const headers = new Headers(init.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  const res = await fetch(`${BASE}${path}`, { ...init, headers })

  if (res.status === 401) throw new UnauthorizedError()
  if (!res.ok) {
    let detail = ''
    try {
      const data = await res.json()
      detail = data.detail ?? ''
    } catch {
      /* ignore */
    }
    throw new Error(detail || `Request failed (${res.status})`)
  }
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export interface UserProfile {
  id: string
  email: string
  display_name: string
  avatar_initials: string
  avatar_gradient: string
}

export async function register(
  email: string,
  password: string,
  displayName: string,
): Promise<UserProfile> {
  return request<UserProfile>('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ email, password, display_name: displayName }),
  })
}

export async function login(email: string, password: string): Promise<string> {
  const data = await request<{ access_token: string }>('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  })
  return data.access_token
}

export async function getMe(): Promise<UserProfile> {
  return request<UserProfile>('/users/me')
}

// ── Projects ────────────────────────────────────────────────────────────────--

export interface Project {
  id: string
  name: string
  icon: string
  created_at: string
  updated_at: string
}

export async function listProjects(): Promise<Project[]> {
  return request<Project[]>('/projects')
}

export async function createProject(name: string, icon?: string): Promise<Project> {
  return request<Project>('/projects', {
    method: 'POST',
    body: JSON.stringify({ name, icon }),
  })
}

export async function renameProject(
  id: string,
  patch: { name?: string; icon?: string },
): Promise<Project> {
  return request<Project>(`/projects/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(patch),
  })
}

export async function deleteProject(id: string): Promise<void> {
  return request<void>(`/projects/${id}`, { method: 'DELETE' })
}

// ── Conversations & messages ───────────────────────────────────────────────---

export interface ConversationSummary {
  id: string
  title: string
  project_id: string | null
  created_at: string
  updated_at: string
}

export interface StoredMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  route: string | null
  sources: string[] | null
  image_url: string | null
  created_at: string
}

export interface ConversationDetail extends ConversationSummary {
  messages: StoredMessage[]
}

export async function listConversations(opts: {
  projectId?: string
  unfiled?: boolean
  limit?: number
} = {}): Promise<ConversationSummary[]> {
  const params = new URLSearchParams()
  if (opts.projectId) params.set('project_id', opts.projectId)
  if (opts.unfiled) params.set('unfiled', 'true')
  if (opts.limit) params.set('limit', String(opts.limit))
  const qs = params.toString()
  return request<ConversationSummary[]>(`/conversations${qs ? `?${qs}` : ''}`)
}

export async function getConversation(id: string): Promise<ConversationDetail> {
  return request<ConversationDetail>(`/conversations/${id}`)
}

export async function createConversation(opts: {
  title?: string
  projectId?: string
} = {}): Promise<ConversationDetail> {
  return request<ConversationDetail>('/conversations', {
    method: 'POST',
    body: JSON.stringify({ title: opts.title, project_id: opts.projectId ?? null }),
  })
}

export async function updateConversation(
  id: string,
  patch: { title?: string; project_id?: string | null },
): Promise<ConversationSummary> {
  return request<ConversationSummary>(`/conversations/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(patch),
  })
}

export async function deleteConversation(id: string): Promise<void> {
  return request<void>(`/conversations/${id}`, { method: 'DELETE' })
}

export interface NewMessage {
  role: 'user' | 'assistant'
  content: string
  route?: string | null
  sources?: string[] | null
  image_url?: string | null
}

export async function appendMessage(
  conversationId: string,
  message: NewMessage,
): Promise<StoredMessage> {
  return request<StoredMessage>(`/conversations/${conversationId}/messages`, {
    method: 'POST',
    body: JSON.stringify(message),
  })
}
