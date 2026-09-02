import { useEffect, useRef } from 'react'
import type { ChatMessage } from '../hooks/useChatStream'

interface ChatPanelProps {
  messages: ChatMessage[]
  isStreaming: boolean
  /** True while there's no usable session to send into yet (e.g. the
   *  initial thread hasn't been created, or creating it failed). */
  disabled?: boolean
  error: string | null
  onSend: (message: string) => void
  inputValue: string
  onInputChange: (value: string) => void
}

export function ChatPanel({
  messages,
  isStreaming,
  disabled = false,
  error,
  onSend,
  inputValue,
  onInputChange,
}: ChatPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const canSubmit = !isStreaming && !disabled

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, isStreaming])

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!inputValue.trim() || !canSubmit) return
    onSend(inputValue)
    onInputChange('')
  }

  return (
    <div className="flex h-full flex-col rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto p-4">
        {messages.length === 0 && !isStreaming && (
          <p className="text-sm text-slate-400">
            Describe your task, or use a quick-start action to auto-fill a prompt…
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[85%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm ${
                m.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-100'
              }`}
            >
              {m.content}
            </div>
          </div>
        ))}
        {isStreaming && (
          <div className="flex justify-start">
            <div className="max-w-[85%] rounded-lg bg-slate-100 px-3 py-2 text-sm text-slate-500 dark:bg-slate-800 dark:text-slate-400">
              …
            </div>
          </div>
        )}
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
            ⚠ {error}
          </div>
        )}
      </div>
      <form onSubmit={handleSubmit} className="flex gap-2 border-t border-slate-200 p-3 dark:border-slate-800">
        <textarea
          value={inputValue}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              handleSubmit(e)
            }
          }}
          placeholder={
            disabled
              ? 'Starting a session…'
              : 'Describe your task, or click a step above to auto-fill…'
          }
          rows={2}
          className="flex-1 resize-none rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
        />
        <button
          type="submit"
          disabled={!canSubmit}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  )
}
