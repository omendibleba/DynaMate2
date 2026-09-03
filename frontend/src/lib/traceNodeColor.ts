// Deterministic pill color classes for an Agent Trace node name — fixed
// colors for the three known system nodes, a stable hash-based color for
// any other (dynamically created specialist agent) name, so a multi-hop
// turn stays visually scannable as more agents are added.

const KNOWN: Record<string, string> = {
  enhancer: 'bg-surface-hover text-ink-muted',
  supervisor: 'bg-brand-50 text-brand-800',
  tool_manager: 'bg-accent-soft text-accent',
}

const PALETTE = [
  'bg-violet-50 text-violet-700 dark:bg-violet-950/40 dark:text-violet-300',
  'bg-amber-50 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300',
  'bg-rose-50 text-rose-700 dark:bg-rose-950/40 dark:text-rose-300',
  'bg-sky-50 text-sky-700 dark:bg-sky-950/40 dark:text-sky-300',
  'bg-fuchsia-50 text-fuchsia-700 dark:bg-fuchsia-950/40 dark:text-fuchsia-300',
]

function hash(str: string): number {
  let h = 0
  for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) | 0
  return Math.abs(h)
}

export function traceNodeColorClasses(node: string): string {
  // parse_chunk() prefixes subgraph-namespaced nodes as "prefix/name" —
  // match/hash on the leaf name so a namespaced supervisor/tool_manager
  // chunk still gets its known color.
  const leaf = node.split('/').pop() ?? node
  return KNOWN[leaf] ?? PALETTE[hash(leaf) % PALETTE.length]
}
