import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { AgentTracePanel } from './components/AgentTracePanel'
import { ChatPanel } from './components/ChatPanel'
import { QuickStartPanel } from './components/QuickStartPanel'
import { StatusSidebar } from './components/StatusSidebar'
import { ThreadHistory } from './components/ThreadHistory'
import { useChatStream } from './hooks/useChatStream'
import { createThread } from './lib/api'

function App() {
  const [threadId, setThreadId] = useState<string | null>(null)
  const [inputValue, setInputValue] = useState('')
  const queryClient = useQueryClient()

  useEffect(() => {
    createThread().then(setThreadId)
  }, [])

  const { messages, trace, isStreaming, error, send, reset } = useChatStream(threadId ?? '', () => {
    queryClient.invalidateQueries({ queryKey: ['status'] })
    queryClient.invalidateQueries({ queryKey: ['threads'] })
  })

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
    <div className="mx-auto flex h-screen max-w-6xl flex-col gap-4 p-4">
      <header className="rounded-xl bg-gradient-to-br from-slate-900 via-blue-950 to-cyan-900 px-6 py-5 text-white">
        <h1 className="text-2xl font-extrabold tracking-tight">⚗️ DynaMate2</h1>
        <p className="text-sm text-cyan-200">Multi-Agent Molecular Simulation Assistant</p>
      </header>

      <QuickStartPanel onSelectPrompt={setInputValue} />

      <div className="grid flex-1 grid-cols-1 gap-4 overflow-hidden md:grid-cols-4">
        <div className="flex flex-col gap-4 overflow-hidden md:col-span-3">
          <div className="flex-1 overflow-hidden">
            <ChatPanel
              messages={messages}
              isStreaming={isStreaming}
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
    </div>
  )
}

export default App
