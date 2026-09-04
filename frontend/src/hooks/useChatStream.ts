import { useCallback, useRef, useState } from 'react'
import { streamChat } from '../lib/api'

export interface TraceEntry {
  node: string
  content: string
  isAi: boolean
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  /** Only set on assistant messages — the full trace for the turn that
   *  produced this reply, so older turns stay inspectable (not just the
   *  most recent one). */
  trace?: TraceEntry[]
}

export function useChatStream(threadId: string, onSent?: () => void, onTrace?: () => void) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  // The in-flight turn's trace, live during streaming — not yet attached
  // to a message (that happens once the final answer arrives, below).
  const [liveTrace, setLiveTrace] = useState<TraceEntry[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const send = useCallback(
    async (message: string) => {
      if (!message.trim() || isStreaming) return

      setMessages((prev) => [...prev, { role: 'user', content: message }])
      setLiveTrace([])
      setError(null)
      setIsStreaming(true)

      const controller = new AbortController()
      abortRef.current = controller

      let finalAnswer = ''
      const turnTrace: TraceEntry[] = []
      try {
        for await (const event of streamChat(threadId, message, controller.signal)) {
          if (event.type === 'trace') {
            const entry: TraceEntry = { node: event.node, content: event.content, isAi: event.is_ai }
            turnTrace.push(entry)
            setLiveTrace((prev) => [...prev, entry])
            // Mirrors app.py's respond(), which recomputed and yielded the
            // status text on every streamed chunk — keeps the Agents & Tools
            // sidebar live during multi-step turns (tool/agent registration)
            // instead of only refreshing once the whole turn finishes.
            onTrace?.()
          } else if (event.type === 'final') {
            finalAnswer = event.answer
          } else if (event.type === 'error') {
            setError(event.message)
          }
        }
      } catch (err) {
        if (!(err instanceof DOMException && err.name === 'AbortError')) {
          setError(err instanceof Error ? err.message : String(err))
        }
      } finally {
        setIsStreaming(false)
        abortRef.current = null
        if (finalAnswer) {
          setMessages((prev) => [...prev, { role: 'assistant', content: finalAnswer, trace: turnTrace }])
        }
        onSent?.()
      }
    },
    [threadId, isStreaming, onSent, onTrace],
  )

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setMessages([])
    setLiveTrace([])
    setError(null)
    setIsStreaming(false)
  }, [])

  return { messages, liveTrace, isStreaming, error, send, reset }
}
