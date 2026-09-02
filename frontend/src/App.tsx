import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { AgentTracePanel } from './components/AgentTracePanel'
import { ChatPanel } from './components/ChatPanel'
import { StatusSidebar } from './components/StatusSidebar'
import { useChatStream } from './hooks/useChatStream'
import { createThread } from './lib/api'

function App() {
  const [threadId, setThreadId] = useState<string | null>(null)
  const [inputValue, setInputValue] = useState('')
  const queryClient = useQueryClient()

  useEffect(() => {
    createThread().then(setThreadId)
  }, [])

  const { messages, trace, isStreaming, error, send } = useChatStream(threadId ?? '', () =>
    queryClient.invalidateQueries({ queryKey: ['status'] }),
  )

  return (
    <div className="mx-auto flex h-screen max-w-6xl flex-col gap-4 p-4">
      <header className="rounded-xl bg-gradient-to-br from-slate-900 via-blue-950 to-cyan-900 px-6 py-5 text-white">
        <h1 className="text-2xl font-extrabold tracking-tight">⚗️ DynaMate2</h1>
        <p className="text-sm text-cyan-200">Multi-Agent Molecular Simulation Assistant</p>
      </header>

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

        <div className="overflow-y-auto md:col-span-1">
          <StatusSidebar />
        </div>
      </div>
    </div>
  )
}

export default App
