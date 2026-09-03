import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Plus } from 'lucide-react'
import { createThread, getThreads } from '../lib/api'

interface ThreadHistoryProps {
  threadId: string | null
  /** New Thread button: starts a fresh thread id and clears the visible chat. */
  onNewThread: (id: string) => void
  /** Resuming from the dropdown: future messages go to this thread id, but
   *  (matching app.py's Gradio behavior) the visible chat pane is untouched —
   *  there's no "replay history" endpoint. */
  onResumeThread: (id: string) => void
}

export function ThreadHistory({ threadId, onNewThread, onResumeThread }: ThreadHistoryProps) {
  const queryClient = useQueryClient()
  const { data: threads } = useQuery({ queryKey: ['threads'], queryFn: getThreads })

  async function handleNewThread() {
    const id = await createThread()
    onNewThread(id)
    queryClient.invalidateQueries({ queryKey: ['threads'] })
  }

  return (
    <div className="rounded-2xl border border-line bg-surface p-4 shadow-sm">
      <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-ink-muted">Session</h2>

      <div className="mb-2 truncate rounded-xl border border-line bg-surface-hover px-2 py-1.5 font-mono text-xs text-ink-muted">
        {threadId ?? '…'}
      </div>

      <button
        onClick={handleNewThread}
        className="mb-3 flex w-full items-center justify-center gap-1.5 rounded-xl bg-surface-hover px-3 py-1.5 text-xs font-semibold text-ink transition hover:bg-line"
      >
        <Plus className="h-3.5 w-3.5" /> New Thread
      </button>

      <label className="mb-1 block text-xs font-medium text-ink-muted">
        Resume a previous thread
      </label>
      <select
        value=""
        onChange={(e) => {
          if (e.target.value) onResumeThread(e.target.value)
        }}
        className="w-full rounded-xl border border-line-strong bg-surface px-2 py-1.5 text-xs text-ink"
      >
        <option value="" disabled>
          Select a thread…
        </option>
        {threads?.map((t) => (
          <option key={t.id} value={t.id}>
            {t.id} — {t.preview}
          </option>
        ))}
      </select>
    </div>
  )
}
