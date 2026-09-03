import { useQueryClient } from '@tanstack/react-query'
import { AlertTriangle, Workflow } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { AgentTracePanel } from './components/AgentTracePanel'
import { ChatPanel } from './components/ChatPanel'
import { NavBar, type TabId } from './components/NavBar'
import { QuickStartPanel } from './components/QuickStartPanel'
import { StatusSidebar } from './components/StatusSidebar'
import { ThreadHistory } from './components/ThreadHistory'
import { useChatStream } from './hooks/useChatStream'
import { createThread } from './lib/api'

function App() {
  const [tab, setTab] = useState<TabId>('chat')
  const [threadId, setThreadId] = useState<string | null>(null)
  const [threadInitError, setThreadInitError] = useState<string | null>(null)
  const [inputValue, setInputValue] = useState('')
  const queryClient = useQueryClient()

  const initThread = useCallback(() => {
    createThread()
      .then((id) => {
        setThreadId(id)
        setThreadInitError(null)
      })
      .catch((err) => setThreadInitError(err instanceof Error ? err.message : String(err)))
  }, [])

  useEffect(() => {
    initThread()
  }, [initThread])

  function handleRetryThread() {
    setThreadInitError(null)
    initThread()
  }

  const { messages, trace, isStreaming, error, send, reset } = useChatStream(
    threadId ?? '',
    () => {
      queryClient.invalidateQueries({ queryKey: ['status'] })
      queryClient.invalidateQueries({ queryKey: ['threads'] })
    },
    () => queryClient.invalidateQueries({ queryKey: ['status'] }),
  )

  function handleNewThread(id: string) {
    setThreadId(id)
    reset()
  }

  function handleResumeThread(id: string) {
    // Matches app.py: switching threads changes where future messages go,
    // but doesn't replay history into the visible chat pane.
    setThreadId(id)
  }

  return (
    <div className="mx-auto flex h-screen max-w-6xl flex-col gap-4 bg-canvas p-4">
      <NavBar tab={tab} onTabChange={setTab} />

      {threadInitError && (
        <div className="flex items-center justify-between rounded-xl border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
          <span className="flex items-center gap-1.5">
            <AlertTriangle className="h-4 w-4" /> Couldn't start a session: {threadInitError}
          </span>
          <button
            onClick={handleRetryThread}
            className="rounded-lg bg-red-100 px-3 py-1 text-xs font-semibold hover:bg-red-200 dark:bg-red-900 dark:hover:bg-red-800"
          >
            Retry
          </button>
        </div>
      )}

      {tab === 'chat' && (
        <div className="grid flex-1 grid-cols-1 gap-4 overflow-hidden md:grid-cols-4">
          <div className="flex flex-col gap-4 overflow-hidden md:col-span-3">
            <div className="flex-1 overflow-hidden">
              <ChatPanel
                messages={messages}
                isStreaming={isStreaming}
                disabled={!threadId}
                error={error}
                onSend={send}
                inputValue={inputValue}
                onInputChange={setInputValue}
              />
            </div>
            <AgentTracePanel trace={trace} />
          </div>

          <div className="flex flex-col gap-4 overflow-y-auto md:col-span-1">
            <StatusSidebar />
            <ThreadHistory
              threadId={threadId}
              onNewThread={handleNewThread}
              onResumeThread={handleResumeThread}
            />
          </div>
        </div>
      )}

      {tab === 'quickstart' && (
        <div className="flex-1 overflow-y-auto">
          <div className="mb-4 rounded-xl border border-line bg-surface p-4">
            <h2 className="text-sm font-semibold text-ink">Quick Start</h2>
            <p className="mt-1 max-w-3xl text-sm text-ink-muted">
              Try DynaMate2 without writing a prompt yourself — these steps mirror the tutorial
              notebook: register tools, spin up a specialist agent, build a simulation box, run
              MD, and plot the trajectory.
            </p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {['MACE-MP-0b3', 'ASE MD', 'Packmol', 'RDKit', 'LangGraph'].map((badge) => (
                <span
                  key={badge}
                  className="rounded-full border border-line-strong bg-surface-hover px-2.5 py-0.5 text-xs text-ink-muted"
                >
                  {badge}
                </span>
              ))}
            </div>
          </div>
          <QuickStartPanel
            onSelectPrompt={(prompt) => {
              setInputValue(prompt)
              setTab('chat')
            }}
          />
        </div>
      )}

      {tab === 'graph' && (
        <div className="flex flex-1 flex-col items-center justify-center gap-2 rounded-xl border border-line bg-surface text-ink-faint">
          <Workflow className="h-8 w-8" />
          <p className="text-sm">Agent graph coming soon…</p>
        </div>
      )}
    </div>
  )
}

export default App
