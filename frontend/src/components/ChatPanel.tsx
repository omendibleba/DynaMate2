import { AlertTriangle, SendHorizontal } from 'lucide-react'
import { Fragment, useEffect, useRef } from 'react'
import type { ChatMessage } from '../hooks/useChatStream'

const FENCE_RE = /```(\w*)\n?([\s\S]*?)```/g
const INLINE_RE = /\*\*(.+?)\*\*|`([^`\n]+?)`/g
const LIST_LINE_RE = /^\s*([-*]|\d+\.)\s+(.*)$/
const MAX_TEXTAREA_HEIGHT = 160

type Variant = 'user' | 'assistant'

/** Splits **bold** and `inline code` spans out of a plain-text run. */
function renderInline(text: string, variant: Variant, keyPrefix: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = []
  let lastIndex = 0
  let key = 0
  for (const match of text.matchAll(INLINE_RE)) {
    const index = match.index ?? 0
    if (index > lastIndex) {
      nodes.push(<Fragment key={`${keyPrefix}-t${key++}`}>{text.slice(lastIndex, index)}</Fragment>)
    }
    if (match[1] !== undefined) {
      nodes.push(<strong key={`${keyPrefix}-b${key++}`}>{match[1]}</strong>)
    } else if (match[2] !== undefined) {
      nodes.push(
        <code
          key={`${keyPrefix}-c${key++}`}
          className={`rounded px-1 py-0.5 font-mono text-[0.85em] ${
            variant === 'user' ? 'bg-white/15' : 'bg-black/10 dark:bg-white/10'
          }`}
        >
          {match[2]}
        </code>,
      )
    }
    lastIndex = index + match[0].length
  }
  if (lastIndex < text.length) nodes.push(<Fragment key={`${keyPrefix}-t${key++}`}>{text.slice(lastIndex)}</Fragment>)
  return nodes
}

/** Groups a plain-text run into paragraph and `- `/`1. ` list blocks, running
 *  each through renderInline() for bold/inline-code. */
function renderTextBlock(text: string, variant: Variant, keyPrefix: string): React.ReactNode[] {
  const blocks: React.ReactNode[] = []
  let paragraphLines: string[] = []
  let listItems: { ordered: boolean; content: string }[] = []
  let key = 0

  function flushParagraph() {
    if (paragraphLines.length === 0) return
    blocks.push(
      <Fragment key={`${keyPrefix}-p${key++}`}>
        {renderInline(paragraphLines.join('\n'), variant, `${keyPrefix}-p${key}`)}
      </Fragment>,
    )
    paragraphLines = []
  }

  function flushList() {
    if (listItems.length === 0) return
    const ordered = listItems[0].ordered
    const Tag = ordered ? 'ol' : 'ul'
    blocks.push(
      <Tag
        key={`${keyPrefix}-l${key++}`}
        className={`my-1 pl-5 ${ordered ? 'list-decimal' : 'list-disc'}`}
        style={{ whiteSpace: 'normal' }}
      >
        {listItems.map((item, i) => (
          <li key={i}>{renderInline(item.content, variant, `${keyPrefix}-li${i}`)}</li>
        ))}
      </Tag>,
    )
    listItems = []
  }

  for (const line of text.split('\n')) {
    const m = LIST_LINE_RE.exec(line)
    if (m) {
      flushParagraph()
      listItems.push({ ordered: /^\d+\.$/.test(m[1]), content: m[2] })
    } else {
      flushList()
      paragraphLines.push(line)
    }
  }
  flushParagraph()
  flushList()
  return blocks
}

/** Splits message text on ```-fenced code blocks (rendered as monospace),
 *  and runs everything else through renderTextBlock() for bold/inline-code/
 *  lists — the patterns that actually show up in agent responses. Returns
 *  React nodes rather than raw HTML: response text is LLM/agent-generated
 *  (semi-trusted), so this avoids ever needing dangerouslySetInnerHTML. */
function renderMessageContent(content: string, variant: Variant) {
  const nodes: React.ReactNode[] = []
  let lastIndex = 0
  let key = 0
  for (const match of content.matchAll(FENCE_RE)) {
    const index = match.index ?? 0
    if (index > lastIndex) {
      nodes.push(...renderTextBlock(content.slice(lastIndex, index), variant, `seg${key++}`))
    }
    const lang = match[1]
    const code = match[2].replace(/\n$/, '')
    nodes.push(
      <pre
        key={`fence${key++}`}
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
  if (lastIndex < content.length) {
    nodes.push(...renderTextBlock(content.slice(lastIndex), variant, `seg${key++}`))
  }
  return nodes
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1 px-0.5 py-1">
      {[0, 150, 300].map((delay) => (
        <span
          key={delay}
          className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-faint"
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
    </div>
  )
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
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const canSubmit = !isStreaming && !disabled

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, isStreaming])

  // Keyed on inputValue (not a textarea onChange handler) so this also
  // fires when a Quick Start action sets the value programmatically
  // (App.tsx's onSelectPrompt -> setInputValue), which never dispatches a
  // native 'input' event.
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT)}px`
  }, [inputValue])

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
              <TypingDots />
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
          ref={textareaRef}
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
          rows={1}
          className="max-h-40 flex-1 resize-none overflow-y-auto rounded-xl border border-line-strong bg-surface px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none"
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
