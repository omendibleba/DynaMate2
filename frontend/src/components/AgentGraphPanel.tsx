import { useQuery } from '@tanstack/react-query'
import mermaid from 'mermaid'
import { Maximize2, ZoomIn, ZoomOut } from 'lucide-react'
import createPanZoom, { type PanZoom } from 'panzoom'
import { useEffect, useId, useRef, useState } from 'react'
import { getStatus } from '../lib/api'
import { buildMermaidGraph } from '../lib/mermaidGraph'

mermaid.initialize({ startOnLoad: false, theme: 'base', securityLevel: 'strict' })

function ZoomButton({ onClick, label, children }: { onClick: () => void; label: string; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      className="flex h-8 w-8 items-center justify-center rounded-lg border border-line bg-surface text-ink-muted shadow-sm transition hover:bg-surface-hover hover:text-ink"
    >
      {children}
    </button>
  )
}

export function AgentGraphPanel() {
  const { data: status, isError } = useQuery({ queryKey: ['status'], queryFn: getStatus })
  const domId = `agent-graph-${useId().replace(/:/g, '')}`
  const [svg, setSvg] = useState('')
  const [renderError, setRenderError] = useState<string | null>(null)

  const viewportRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const panzoomRef = useRef<PanZoom | null>(null)

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

  // Set up once — dangerouslySetInnerHTML above only replaces this node's
  // children on re-render, so the panzoom-applied transform (set directly
  // on this same element's style) survives graph content updates.
  useEffect(() => {
    if (!contentRef.current) return
    const instance = createPanZoom(contentRef.current, {
      maxZoom: 4,
      minZoom: 0.2,
      zoomSpeed: 0.065,
      bounds: false,
      smoothScroll: false,
    })
    panzoomRef.current = instance
    return () => {
      instance.dispose()
      panzoomRef.current = null
    }
  }, [])

  function zoomAtCenter(scaleMultiplier: number) {
    const instance = panzoomRef.current
    const viewport = viewportRef.current
    if (!instance || !viewport) return
    const rect = viewport.getBoundingClientRect()
    instance.smoothZoom(rect.width / 2, rect.height / 2, scaleMultiplier)
  }

  function resetView() {
    const instance = panzoomRef.current
    if (!instance) return
    instance.moveTo(0, 0)
    instance.zoomAbs(0, 0, 1)
  }

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

      <div ref={viewportRef} className="relative flex-1 overflow-hidden">
        <div ref={contentRef} className="h-full w-full origin-top-left" dangerouslySetInnerHTML={{ __html: svg }} />
        <div className="absolute bottom-3 right-3 flex flex-col gap-1.5">
          <ZoomButton onClick={() => zoomAtCenter(1.4)} label="Zoom in">
            <ZoomIn className="h-4 w-4" />
          </ZoomButton>
          <ZoomButton onClick={() => zoomAtCenter(1 / 1.4)} label="Zoom out">
            <ZoomOut className="h-4 w-4" />
          </ZoomButton>
          <ZoomButton onClick={resetView} label="Reset view">
            <Maximize2 className="h-4 w-4" />
          </ZoomButton>
        </div>
      </div>
    </div>
  )
}
