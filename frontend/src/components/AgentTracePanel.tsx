import { ListTree } from 'lucide-react'
import { useEffect, useRef } from 'react'
import type { ChatMessage, TraceEntry } from '../hooks/useChatStream'
import { traceNodeColorClasses } from '../lib/traceNodeColor'

interface AgentTracePanelProps {
  messages: ChatMessage[]
  liveTrace: TraceEntry[]
  isStreaming?: boolean
}

interface Turn {
  key: number
  userPreview: string
  trace: TraceEntry[]
  live: boolean
}

const PREVIEW_MAX = 60

/** Pairs each user message with the trace of the reply that followed it —
 *  a user message with no assistant reply yet is the in-flight turn, shown
 *  from liveTrace instead of a finalized message.trace. */
function buildTurns(messages: ChatMessage[], liveTrace: TraceEntry[], isStreaming: boolean): Turn[] {
  const turns: Turn[] = []
  let key = 0
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i]
    if (m.role !== 'user') continue
    const preview = m.content.length > PREVIEW_MAX ? `${m.content.slice(0, PREVIEW_MAX)}…` : m.content
    const next = messages[i + 1]
    if (next?.role === 'assistant') {
      turns.push({ key: key++, userPreview: preview, trace: next.trace ?? [], live: false })
    } else {
      turns.push({ key: key++, userPreview: preview, trace: liveTrace, live: isStreaming })
    }
  }
  return turns
}

export function AgentTracePanel({ messages, liveTrace, isStreaming = false }: AgentTracePanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const turns = buildTurns(messages, liveTrace, isStreaming)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, liveTrace])

  return (
    <div className="flex h-full flex-col rounded-2xl border border-line bg-surface shadow-sm">
      <div className="flex items-center gap-2 border-b border-line px-4 py-2.5 text-sm font-semibold text-ink-muted">
        <ListTree className="h-4 w-4" />
        <span>Agent Trace</span>
        {isStreaming && <span className="h-2 w-2 animate-pulse rounded-full bg-accent" />}
      </div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-3 font-mono text-xs text-ink-muted">
        {turns.length === 0 ? (
          <p className="text-ink-faint">Routing steps and tool calls will appear here…</p>
        ) : (
          turns.map((turn) => (
            <div key={turn.key} className="mb-3">
              <div className="mb-1.5 flex items-center gap-1.5 border-b border-line pb-1 font-sans text-[11px] font-semibold text-ink">
                <span className="text-ink-faint">#{turn.key + 1}</span>
                <span className="truncate">{turn.userPreview}</span>
                {turn.live && <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-accent" />}
              </div>
              {turn.trace.length === 0 ? (
                <p className="text-ink-faint">
                  {turn.live ? 'Routing…' : 'No trace recorded for this turn.'}
                </p>
              ) : (
                turn.trace.map((t, i) => (
                  <div key={i} className="mb-1.5">
                    <span
                      className={`mr-1.5 rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${traceNodeColorClasses(t.node)}`}
                    >
                      {t.node}
                    </span>
                    {t.content}
                  </div>
                ))
              )}
            </div>
          ))
        )}
      </div>
    </div>
  )
}
