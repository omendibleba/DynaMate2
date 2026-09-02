import { useQuery, useQueryClient } from '@tanstack/react-query'
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
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">
        Session
      </h2>

      <div className="mb-2 truncate rounded-lg border border-slate-200 bg-slate-50 px-2 py-1.5 font-mono text-xs text-slate-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-400">
        {threadId ?? '…'}
      </div>

      <button
        onClick={handleNewThread}
        className="mb-3 w-full rounded-lg bg-slate-100 px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
      >
        ＋ New Thread
      </button>

      <label className="mb-1 block text-xs font-medium text-slate-500 dark:text-slate-400">
        Resume a previous thread
      </label>
      <select
        value=""
        onChange={(e) => {
          if (e.target.value) onResumeThread(e.target.value)
        }}
        className="w-full rounded-lg border border-slate-300 px-2 py-1.5 text-xs dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100"
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
