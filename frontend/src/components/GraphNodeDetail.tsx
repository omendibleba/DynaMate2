import { Bot, Wrench, X } from 'lucide-react'
import type { GraphNode } from '../lib/mermaidGraph'

interface GraphNodeDetailProps {
  node: GraphNode
  description: string
  onClose: () => void
}

export function GraphNodeDetail({ node, description, onClose }: GraphNodeDetailProps) {
  const isAgent = node.kind === 'agent'

  return (
    <div className="absolute right-3 top-3 max-h-[70%] w-80 overflow-y-auto rounded-xl border border-line bg-surface p-3 shadow-md">
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5 text-sm font-semibold text-ink">
          {isAgent ? <Bot className="h-4 w-4 shrink-0" /> : <Wrench className="h-4 w-4 shrink-0" />}
          <span className="break-words">{node.name}</span>
        </div>
        <button
          onClick={onClose}
          aria-label="Close"
          className="shrink-0 rounded-md p-0.5 text-ink-faint transition hover:bg-surface-hover hover:text-ink"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      {!isAgent && <p className="mb-2 text-xs text-ink-faint">Tool of {node.agentName}</p>}
      {description.trim() ? (
        <p className="whitespace-pre-wrap text-xs text-ink-muted">{description}</p>
      ) : (
        <p className="text-xs text-ink-faint">No description available.</p>
      )}
    </div>
  )
}
