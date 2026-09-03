import { ListTree } from 'lucide-react'
import type { TraceEntry } from '../hooks/useChatStream'

interface AgentTracePanelProps {
  trace: TraceEntry[]
  isStreaming?: boolean
}

export function AgentTracePanel({ trace, isStreaming = false }: AgentTracePanelProps) {
  return (
    <div className="flex h-full flex-col rounded-2xl border border-line bg-surface shadow-sm">
      <div className="flex items-center gap-2 border-b border-line px-4 py-2.5 text-sm font-semibold text-ink-muted">
        <ListTree className="h-4 w-4" />
        <span>Agent Trace</span>
        {isStreaming && <span className="h-2 w-2 animate-pulse rounded-full bg-accent" />}
      </div>
      <div className="flex-1 overflow-y-auto p-3 font-mono text-xs text-ink-muted">
        {trace.length === 0 ? (
          <p className="text-ink-faint">Routing steps and tool calls will appear here…</p>
        ) : (
          trace.map((t, i) => (
            <div key={i} className="mb-1.5">
              <span className="mr-1.5 rounded-full bg-brand-50 px-1.5 py-0.5 text-[10px] font-semibold text-brand-800">
                {t.node}
              </span>
              {t.content}
            </div>
          ))
        )}
      </div>
    </div>
  )
}
