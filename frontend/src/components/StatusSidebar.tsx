import { useQuery } from '@tanstack/react-query'
import { getStatus } from '../lib/api'

export function StatusSidebar() {
  const { data, isLoading, refetch } = useQuery({
    queryKey: ['status'],
    queryFn: getStatus,
  })

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">
          Agents &amp; Tools
        </h2>
        <button
          onClick={() => refetch()}
          className="text-xs font-medium text-blue-600 hover:underline dark:text-blue-400"
        >
          ↻ Refresh
        </button>
      </div>

      {isLoading && <p className="text-sm text-slate-400">Loading…</p>}

      {data && (
        <div className="space-y-3 text-sm">
          {data.agents.map((agent) => (
            <div key={agent.name}>
              <div className="font-semibold text-slate-700 dark:text-slate-200">{agent.name}</div>
              {agent.base_tools.length > 0 && (
                <div className="text-xs text-slate-500 dark:text-slate-400">
                  base: {agent.base_tools.join(', ')}
                </div>
              )}
              {agent.extra_tools.length > 0 && (
                <div className="text-xs text-slate-500 dark:text-slate-400">
                  tools: {agent.extra_tools.join(', ')}
                </div>
              )}
            </div>
          ))}

          <div className="pt-2">
            <div className="font-semibold text-slate-700 dark:text-slate-200">Registry</div>
            {data.registry.length === 0 ? (
              <div className="text-xs text-slate-400">(empty)</div>
            ) : (
              <ul className="list-inside list-disc text-xs text-slate-500 dark:text-slate-400">
                {data.registry.map((name) => (
                  <li key={name}>{name}</li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
