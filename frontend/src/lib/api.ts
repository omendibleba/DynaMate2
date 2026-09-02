// Typed client for the DynaMate2 FastAPI backend (backend/). Mirrors
// backend/schemas.py.

export interface AgentStatus {
  name: string
  base_tools: string[]
  extra_tools: string[]
}

export interface StatusResponse {
  agents: AgentStatus[]
  registry: string[]
}

export interface ThreadInfo {
  id: string
  preview: string
  created_at: string
}

export interface QuickstartPrompts {
  t1a: string
  t1b: string
  t1c: string
  t2: string
  t3a: string
  t3b: string
  t4a: string
  t4b: string
}

export interface UploadResponse {
  path: string
  prompt: string
}

export type ChatStreamEvent =
  | { type: 'trace'; node: string; content: string; is_ai: boolean }
  | { type: 'final'; answer: string }
  | { type: 'error'; message: string }

async function getJSON<T>(path: string): Promise<T> {
  const resp = await fetch(path)
  if (!resp.ok) throw new Error(`${path} -> ${resp.status}`)
  return resp.json() as Promise<T>
}

export function getStatus(): Promise<StatusResponse> {
  return getJSON<StatusResponse>('/api/status')
}

export function getThreads(): Promise<ThreadInfo[]> {
  return getJSON<ThreadInfo[]>('/api/threads')
}

export async function createThread(): Promise<string> {
  const resp = await fetch('/api/threads', { method: 'POST' })
  if (!resp.ok) throw new Error(`POST /api/threads -> ${resp.status}`)
  const body = (await resp.json()) as { id: string }
  return body.id
}

export function getQuickstartPrompts(): Promise<QuickstartPrompts> {
  return getJSON<QuickstartPrompts>('/api/quickstart/prompts')
}

export async function uploadTool(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  const resp = await fetch('/api/tools/upload', { method: 'POST', body: formData })
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ detail: resp.statusText }))
    throw new Error(body.detail ?? `POST /api/tools/upload -> ${resp.status}`)
  }
  return resp.json() as Promise<UploadResponse>
}

/**
 * Streams a chat turn as an async generator of parsed SSE events. Browser
 * EventSource can't send a POST body, so this reads the response body as a
 * ReadableStream and parses `event:`/`data:` frames by hand — still
 * standard SSE wire format, just consumed via fetch instead of
 * `new EventSource()`.
 */
export async function* streamChat(
  threadId: string,
  message: string,
  signal?: AbortSignal,
): AsyncGenerator<ChatStreamEvent> {
  const resp = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ thread_id: threadId, message }),
    signal,
  })
  if (!resp.ok || !resp.body) {
    throw new Error(`POST /api/chat/stream -> ${resp.status}`)
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  function parseFrame(frame: string): ChatStreamEvent | null {
    let eventType = 'message'
    let data = ''
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) eventType = line.slice(6).trim()
      else if (line.startsWith('data:')) data = line.slice(5).trim()
    }
    if (!data) return null
    return { type: eventType, ...JSON.parse(data) } as ChatStreamEvent
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    // Normalize line endings: some proxies/relays (e.g. tunnel HTTP relays)
    // rewrite \n to \r\n in transit, which would otherwise silently break
    // the \n\n frame-boundary search below.
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')

    // SSE frames are separated by a blank line.
    let sep: number
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const event = parseFrame(frame)
      if (event) yield event
    }
  }

  // Defensive: process a final frame even if the stream ended without a
  // trailing blank line (shouldn't happen with a well-behaved SSE server,
  // but a lossy intermediary could still truncate the last separator).
  const remaining = buffer.trim()
  if (remaining) {
    const event = parseFrame(remaining)
    if (event) yield event
  }
}
