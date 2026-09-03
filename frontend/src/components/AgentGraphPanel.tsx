import { useQuery } from '@tanstack/react-query'
import mermaid from 'mermaid'
import { useEffect, useId, useState } from 'react'
import { getStatus } from '../lib/api'
import { buildMermaidGraph } from '../lib/mermaidGraph'

mermaid.initialize({ startOnLoad: false, theme: 'base', securityLevel: 'strict' })

export function AgentGraphPanel() {
  const { data: status, isError } = useQuery({ queryKey: ['status'], queryFn: getStatus })
  const domId = `agent-graph-${useId().replace(/:/g, '')}`
  const [svg, setSvg] = useState('')
  const [renderError, setRenderError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const definition = buildMermaidGraph(status ?? { agents: [], registry: [] })
    mermaid
      .render(domId, definition)
      .then((result) => {
        if (!cancelled) {
          setSvg(result.svg)
          setRenderError(null)
        }
      })
      .catch((err) => {
        if (!cancelled) setRenderError(err instanceof Error ? err.message : String(err))
      })
    return () => {
      cancelled = true
    }
  }, [status, domId])

  return (
    <div className="flex h-full flex-col rounded-xl border border-line bg-surface p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-wide text-ink-muted">Agent Graph</h2>
        <div className="flex items-center gap-3 text-xs text-ink-faint">
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-brand-300" /> base tool
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-accent" /> registered at runtime
          </span>
        </div>
      </div>

      {isError && (
        <p className="text-sm text-ink-faint">Couldn't load agent status — showing an empty graph.</p>
      )}
      {renderError && <p className="text-sm text-ink-faint">Couldn't render graph: {renderError}</p>}

      <div className="flex-1 overflow-auto" dangerouslySetInnerHTML={{ __html: svg }} />
    </div>
  )
}
