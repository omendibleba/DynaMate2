import { useState } from 'react'
import type { TraceEntry } from '../hooks/useChatStream'

export function AgentTracePanel({ trace }: { trace: TraceEntry[] }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-4 py-2 text-sm font-semibold text-slate-600 dark:text-slate-300"
      >
        <span>🔍 Agent trace</span>
        <span>{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="max-h-64 overflow-y-auto border-t border-slate-200 p-3 font-mono text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
          {trace.length === 0 ? (
            <p>Routing steps and tool calls will appear here…</p>
          ) : (
            trace.map((t, i) => (
              <div key={i} className="mb-1">
                [{t.node}] {t.content}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  )
}
