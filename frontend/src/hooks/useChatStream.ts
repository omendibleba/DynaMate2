import { useCallback, useRef, useState } from 'react'
import { streamChat } from '../lib/api'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface TraceEntry {
  node: string
  content: string
  isAi: boolean
}

export function useChatStream(threadId: string, onSent?: () => void, onTrace?: () => void) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [trace, setTrace] = useState<TraceEntry[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const send = useCallback(
    async (message: string) => {
      if (!message.trim() || isStreaming) return

      setMessages((prev) => [...prev, { role: 'user', content: message }])
      setTrace([])
      setError(null)
      setIsStreaming(true)

      const controller = new AbortController()
      abortRef.current = controller

      let finalAnswer = ''
      try {
        for await (const event of streamChat(threadId, message, controller.signal)) {
          if (event.type === 'trace') {
            setTrace((prev) => [...prev, { node: event.node, content: event.content, isAi: event.is_ai }])
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
          setMessages((prev) => [...prev, { role: 'assistant', content: finalAnswer }])
        }
        onSent?.()
      }
    },
    [threadId, isStreaming, onSent, onTrace],
  )

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setMessages([])
    setTrace([])
    setError(null)
    setIsStreaming(false)
  }, [])

  return { messages, trace, isStreaming, error, send, reset }
}
