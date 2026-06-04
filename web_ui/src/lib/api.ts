/**
 * Typed client for the cortex_api AI service.
 *
 * All calls go through the relative `/api` path, which Vite proxies to the
 * cortex_api server (http://localhost:8001 by default — see vite.config.ts).
 */

const BASE = '/api'

// ── Chat (SSE streaming) ──────────────────────────────────────────────────────

export interface ChatDone {
  route: string
  scores: Record<string, number>
  sources: string[]
}

export interface ChatHandlers {
  onToken: (token: string) => void
  onDone: (info: ChatDone) => void
  onError: (message: string) => void
  signal?: AbortSignal
}

/**
 * Stream a chat answer. Resolves when the stream ends; tokens arrive via
 * `handlers.onToken`, and routing metadata via `handlers.onDone`.
 */
export async function chatStream(query: string, handlers: ChatHandlers): Promise<void> {
  let res: Response
  try {
    res = await fetch(`${BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
      signal: handlers.signal,
    })
  } catch (err) {
    handlers.onError(err instanceof Error ? err.message : 'Network error')
    return
  }

  if (!res.ok || !res.body) {
    handlers.onError(`Request failed (${res.status})`)
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE events are separated by a blank line.
    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''

    for (const evt of events) {
      const line = evt.split('\n').find((l) => l.startsWith('data:'))
      if (!line) continue
      const payload = line.slice(5).trim()
      if (!payload) continue
      let msg: { type: string; value?: string; message?: string } & Partial<ChatDone>
      try {
        msg = JSON.parse(payload)
      } catch {
        continue
      }
      if (msg.type === 'token' && msg.value !== undefined) {
        handlers.onToken(msg.value)
      } else if (msg.type === 'done') {
        handlers.onDone({
          route: msg.route ?? 'chat',
          scores: msg.scores ?? {},
          sources: msg.sources ?? [],
        })
      } else if (msg.type === 'error') {
        handlers.onError(msg.message ?? 'Unknown error')
      }
    }
  }
}

// ── Documents ─────────────────────────────────────────────────────────────────

export interface DocumentInfo {
  name: string
  type: string
  size: string
  updated: string
}

export interface DocumentsResponse {
  documents: DocumentInfo[]
  indexedCount: number
}

export async function listDocuments(): Promise<DocumentsResponse> {
  const res = await fetch(`${BASE}/documents`)
  if (!res.ok) throw new Error(`Failed to list documents (${res.status})`)
  return res.json()
}

export interface UploadResult {
  name: string
  chunksAdded: number
}

export async function uploadDocument(file: File): Promise<UploadResult> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/documents`, { method: 'POST', body: form })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(`Upload failed (${res.status}) ${detail}`)
  }
  return res.json()
}

// ── Search ────────────────────────────────────────────────────────────────────

export interface SearchHit {
  source: string
  snippet: string
  score: number
  page?: string
}

export async function search(query: string, k = 5): Promise<SearchHit[]> {
  const res = await fetch(`${BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, k }),
  })
  if (!res.ok) throw new Error(`Search failed (${res.status})`)
  const data = await res.json()
  return data.results ?? []
}

// ── Stats (Overview dashboard) ────────────────────────────────────────────────

export interface VolumePoint {
  name: string
  date: string
  queries: number
  rag: number
}

export interface ActivityItem {
  id: string
  type: 'info' | 'success' | 'warning'
  message: string
  time: string
}

export interface Stats {
  documentsIndexed: number
  queriesToday: number
  queriesYesterday: number
  queriesLast7Days: number
  ragQueries: number
  chatQueries: number
  routerAccuracy: number | null
  totalPredictions: number
  volumeSeries: VolumePoint[]
  activity: ActivityItem[]
}

/** Fetch real Overview metrics from the cortex_api service. */
export async function getStats(): Promise<Stats> {
  const res = await fetch(`${BASE}/stats`)
  if (!res.ok) throw new Error(`Failed to load stats (${res.status})`)
  return res.json()
}

// ── Voice (STT / TTS) ─────────────────────────────────────────────────────────

/** Transcribe recorded audio to text via Parakeet. */
export async function stt(audio: Blob): Promise<string> {
  const form = new FormData()
  form.append('file', audio, 'recording.webm')
  const res = await fetch(`${BASE}/stt`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Transcription failed (${res.status})`)
  const data = await res.json()
  return data.text ?? ''
}

/** Synthesize speech via Coqui; returns a playable object URL. */
export async function tts(text: string): Promise<string> {
  const res = await fetch(`${BASE}/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) throw new Error(`Speech synthesis failed (${res.status})`)
  const blob = await res.blob()
  return URL.createObjectURL(blob)
}

// ── Vision (VLM) ────────────────────────────────────────────────────────────---

export async function vlm(image: File, prompt: string): Promise<string> {
  const form = new FormData()
  form.append('image', image)
  form.append('prompt', prompt)
  const res = await fetch(`${BASE}/vlm`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Vision request failed (${res.status})`)
  const data = await res.json()
  return data.description ?? ''
}
