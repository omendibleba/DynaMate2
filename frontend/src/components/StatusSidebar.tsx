import { useQuery } from '@tanstack/react-query'
import { RefreshCw } from 'lucide-react'
import { getStatus } from '../lib/api'

function ToolPill({ name, variant }: { name: string; variant: 'base' | 'extra' }) {
  return (
    <span
      className={`rounded-full border px-2 py-0.5 text-[11px] ${
        variant === 'base'
          ? 'border-brand-200 bg-brand-50 text-brand-800'
          : 'border-accent/40 bg-accent-soft text-accent'
      }`}
    >
      {name}
    </span>
  )
}

export function StatusSidebar() {
  const { data, isLoading, refetch } = useQuery({
    queryKey: ['status'],
    queryFn: getStatus,
  })

  return (
    <div className="rounded-2xl border border-line bg-surface p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-wide text-ink-muted">Agents &amp; Tools</h2>
        <button
          onClick={() => refetch()}
          className="flex items-center gap-1 text-xs font-medium text-brand-600 hover:underline"
        >
          <RefreshCw className="h-3.5 w-3.5" /> Refresh
        </button>
      </div>

      {isLoading && <p className="text-sm text-ink-faint">Loading…</p>}

      {data && (
        <div className="space-y-3 text-sm">
          {data.agents.map((agent) => (
            <div key={agent.name}>
              <div className="font-semibold text-ink">{agent.name}</div>
              {(agent.base_tools.length > 0 || agent.extra_tools.length > 0) && (
                <div className="mt-1 flex flex-wrap gap-1">
                  {agent.base_tools.map((tool) => (
                    <ToolPill key={tool} name={tool} variant="base" />
                  ))}
                  {agent.extra_tools.map((tool) => (
                    <ToolPill key={tool} name={tool} variant="extra" />
                  ))}
                </div>
              )}
            </div>
          ))}

          <div className="pt-2">
            <div className="font-semibold text-ink">Registry</div>
            {data.registry.length === 0 ? (
              <div className="text-xs text-ink-faint">(empty)</div>
            ) : (
              <div className="mt-1 flex flex-wrap gap-1">
                {data.registry.map((name) => (
                  <span
                    key={name}
                    className="rounded-full border border-line-strong bg-surface-hover px-2 py-0.5 text-[11px] text-ink-muted"
                  >
                    {name}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
