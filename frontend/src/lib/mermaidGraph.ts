// Builds a mermaid flowchart definition from the pool status that
// GET /api/status already returns. LangGraph's own graph
// (pool.supervisor.get_graph()) only has supervisor + agent-level nodes —
// tools are bound into each agent's internal ReAct loop and never appear
// as graph nodes — so this is synthesized client-side instead, using the
// same data StatusSidebar already renders. Node ids are always positional
// (never derived from raw names) since tool/agent names can contain
// characters (spaces, parens, underscores) that break mermaid syntax.

import type { StatusResponse } from './api'

// Keep these hex values in sync with the brand/accent tokens in index.css —
// mermaid renders to a detached SVG string and can't consume CSS custom
// properties.
const CLASS_DEFS = [
  'classDef supervisor fill:#0b2c4f,stroke:#0b2c4f,color:#ffffff',
  'classDef agent fill:#2569a3,stroke:#0b2c4f,color:#ffffff',
  'classDef base fill:#eef5fc,stroke:#2569a3,color:#0b2c4f',
  'classDef extra fill:#e6f7f5,stroke:#0e9f8e,color:#0b6b60',
].join('\n')

function escapeLabel(name: string): string {
  return name.replace(/"/g, '&quot;')
}

export type GraphNode = { kind: 'agent'; name: string } | { kind: 'tool'; name: string; agentName: string }

/** Looks a clicked node back up by its rendered LABEL TEXT, not a DOM id —
 *  mermaid's generated id format isn't part of its public contract and
 *  varies by renderer/version, but the label text we provide is rendered
 *  verbatim, and the :::agent/:::base/:::extra classDef names we already
 *  use for node color ARE guaranteed to land as real CSS classes on the
 *  rendered node (that's the only way the color-coding works at all). */
export interface NodeIndex {
  agents: Record<string, GraphNode>
  tools: Record<string, GraphNode>
}

export function buildMermaidGraph(status: StatusResponse): { definition: string; nodeIndex: NodeIndex } {
  const lines: string[] = ['flowchart LR', 'supervisor(["Supervisor"]):::supervisor']
  const nodeIndex: NodeIndex = { agents: {}, tools: {} }

  status.agents.forEach((agent, i) => {
    const agentId = `agent_${i}`
    lines.push(`${agentId}["${escapeLabel(agent.name)}"]:::agent`)
    lines.push(`supervisor --> ${agentId}`)
    nodeIndex.agents[agent.name] = { kind: 'agent', name: agent.name }

    agent.base_tools.forEach((tool, j) => {
      const toolId = `${agentId}_base_${j}`
      lines.push(`${toolId}["${escapeLabel(tool)}"]:::base`)
      lines.push(`${agentId} --- ${toolId}`)
      nodeIndex.tools[tool] = { kind: 'tool', name: tool, agentName: agent.name }
    })

    agent.extra_tools.forEach((tool, j) => {
      const toolId = `${agentId}_extra_${j}`
      lines.push(`${toolId}["${escapeLabel(tool)}"]:::extra`)
      lines.push(`${agentId} --- ${toolId}`)
      nodeIndex.tools[tool] = { kind: 'tool', name: tool, agentName: agent.name }
    })
  })

  lines.push(CLASS_DEFS)
  return { definition: lines.join('\n'), nodeIndex }
}
