import { AlertTriangle, SendHorizontal } from 'lucide-react'
import { Fragment, useEffect, useRef } from 'react'
import type { ChatMessage } from '../hooks/useChatStream'

const FENCE_RE = /```(\w*)\n?([\s\S]*?)```/g

/** Splits message text on ```-fenced code blocks and renders each fence as
 *  a monospace block, matching how tool code (e.g. the LLM-registered-tool
 *  quick-start flow) actually appears in chat responses. */
function renderMessageContent(content: string, variant: 'user' | 'assistant') {
  const nodes: React.ReactNode[] = []
  let lastIndex = 0
  let key = 0
  for (const match of content.matchAll(FENCE_RE)) {
    const index = match.index ?? 0
    if (index > lastIndex) nodes.push(<Fragment key={key++}>{content.slice(lastIndex, index)}</Fragment>)
    const lang = match[1]
    const code = match[2].replace(/\n$/, '')
    nodes.push(
      <pre
        key={key++}
        className={`my-1.5 overflow-x-auto rounded-lg p-2 font-mono text-xs ${
          variant === 'user'
            ? 'border border-white/15 bg-white/10 text-white'
            : 'border border-line bg-canvas text-ink'
        }`}
      >
        {lang && <div className="mb-1 text-[10px] uppercase tracking-wide opacity-60">{lang}</div>}
        <code>{code}</code>
      </pre>,
    )
    lastIndex = index + match[0].length
  }
  if (lastIndex < content.length) nodes.push(<Fragment key={key++}>{content.slice(lastIndex)}</Fragment>)
  return nodes
}

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
    <div className="flex h-full flex-col rounded-2xl border border-line bg-surface shadow-sm">
      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto p-4">
        {messages.length === 0 && !isStreaming && (
          <p className="text-sm text-ink-faint">
            Describe your task, or use a quick-start action to auto-fill a prompt…
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[85%] whitespace-pre-wrap rounded-2xl px-3.5 py-2 text-sm ${
                m.role === 'user' ? 'bg-brand-800 text-white' : 'bg-surface-hover text-ink'
              }`}
            >
              {renderMessageContent(m.content, m.role)}
            </div>
          </div>
        ))}
        {isStreaming && (
          <div className="flex justify-start">
            <div className="max-w-[85%] rounded-2xl bg-surface-hover px-3.5 py-2 text-sm text-ink-muted">
              …
            </div>
          </div>
        )}
        {error && (
          <div className="flex items-center gap-1.5 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
            <AlertTriangle className="h-4 w-4 shrink-0" /> {error}
          </div>
        )}
      </div>
      <form onSubmit={handleSubmit} className="flex gap-2 border-t border-line p-3">
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
          className="flex-1 resize-none rounded-xl border border-line-strong bg-surface px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none"
        />
        <button
          type="submit"
          disabled={!canSubmit}
          className="flex items-center gap-1.5 rounded-xl bg-brand-800 px-4 py-2 text-sm font-semibold text-white transition hover:bg-brand-900 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <SendHorizontal className="h-4 w-4" />
          Send
        </button>
      </form>
    </div>
  )
}
