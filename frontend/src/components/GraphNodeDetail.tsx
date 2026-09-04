import { useQueryClient } from '@tanstack/react-query'
import { Bot, Check, Pencil, Wrench, X } from 'lucide-react'
import { useState } from 'react'
import { updateAgentPrompt, updateToolDescription } from '../lib/api'
import type { GraphNode } from '../lib/mermaidGraph'

interface GraphNodeDetailProps {
  node: GraphNode
  description: string
  onClose: () => void
}

export function GraphNodeDetail({ node, description, onClose }: GraphNodeDetailProps) {
  const isAgent = node.kind === 'agent'
  const queryClient = useQueryClient()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(description)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)

  function startEditing() {
    setDraft(description)
    setSaveError(null)
    setEditing(true)
  }

  async function handleSave() {
    setSaving(true)
    setSaveError(null)
    try {
      if (isAgent) {
        await updateAgentPrompt(node.name, draft)
      } else {
        await updateToolDescription(node.name, draft)
      }
      // The supervisor and every agent holding this tool are rebuilt
      // server-side (dynamate/pool.py's update_agent_prompt/
      // update_tool_description) — refetching status is enough to pick up
      // the new text here, no extra plumbing needed.
      await queryClient.invalidateQueries({ queryKey: ['status'] })
      setEditing(false)
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div
      className="absolute right-3 top-3 max-h-[70%] w-80 overflow-y-auto rounded-xl border border-line bg-surface p-3 shadow-md"
      // panzoom listens on `document` for mouseup/click (not just the graph
      // canvas), so a click anywhere on the page — including this panel,
      // which sits as a sibling overlay, not a descendant, of the
      // panzoom-controlled element — can still reach it and get
      // misinterpreted as a graph click that resolves to no node,
      // clearing the selection. Stop these before they bubble that far.
      onMouseDown={(e) => e.stopPropagation()}
      onMouseUp={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
    >
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5 text-sm font-semibold text-ink">
          {isAgent ? <Bot className="h-4 w-4 shrink-0" /> : <Wrench className="h-4 w-4 shrink-0" />}
          <span className="break-words">{node.name}</span>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {!editing && (
            <button
              onClick={startEditing}
              aria-label="Edit description"
              className="rounded-md p-0.5 text-ink-faint transition hover:bg-surface-hover hover:text-ink"
            >
              <Pencil className="h-3.5 w-3.5" />
            </button>
          )}
          <button
            onClick={onClose}
            aria-label="Close"
            className="rounded-md p-0.5 text-ink-faint transition hover:bg-surface-hover hover:text-ink"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      {!isAgent && <p className="mb-2 text-xs text-ink-faint">Tool of {node.agentName}</p>}

      {editing ? (
        <div className="space-y-1.5">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={8}
            autoFocus
            className="w-full resize-y rounded-lg border border-line-strong bg-surface p-2 font-mono text-xs text-ink focus:border-accent focus:outline-none"
          />
          <p className="text-[10px] text-ink-faint">
            {isAgent
              ? 'Takes effect immediately. Persists across restarts for agents created at runtime.'
              : 'Takes effect immediately for this session — resets to the source docstring on restart.'}
          </p>
          {saveError && <p className="text-xs text-red-600 dark:text-red-400">{saveError}</p>}
          <div className="flex justify-end gap-2 pt-0.5">
            <button
              onClick={() => setEditing(false)}
              disabled={saving}
              className="rounded-lg px-2.5 py-1 text-xs font-medium text-ink-muted transition hover:bg-surface-hover disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={saving || !draft.trim()}
              className="flex items-center gap-1 rounded-lg bg-brand-800 px-2.5 py-1 text-xs font-semibold text-white transition hover:bg-brand-900 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Check className="h-3.5 w-3.5" />
              {saving ? 'Saving…' : 'Save'}
            </button>
          </div>
        </div>
      ) : description.trim() ? (
        <p className="whitespace-pre-wrap text-xs text-ink-muted">{description}</p>
      ) : (
        <p className="text-xs text-ink-faint">No description available.</p>
      )}
    </div>
  )
}
