import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { getQuickstartPrompts } from '../lib/api'
import { FileUploadZone } from './FileUploadZone'

interface QuickStartPanelProps {
  onSelectPrompt: (prompt: string) => void
}

type TabId = 'prompt' | 'script' | 'llm' | 'run'

const TABS: { id: TabId; label: string }[] = [
  { id: 'prompt', label: '📋 From Prompt' },
  { id: 'script', label: '📄 From Script' },
  { id: 'llm', label: '🤖 From LLM' },
  { id: 'run', label: '▶ Run Simulations' },
]

function StepButton({
  label,
  hint,
  colorClass,
  onClick,
}: {
  label: string
  hint: string
  colorClass: string
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full rounded-lg border px-3 py-2 text-left text-sm font-semibold transition ${colorClass}`}
    >
      <div>{label}</div>
      <div className="mt-0.5 text-xs font-normal opacity-70">{hint}</div>
    </button>
  )
}

export function QuickStartPanel({ onSelectPrompt }: QuickStartPanelProps) {
  const [tab, setTab] = useState<TabId>('prompt')
  const { data: prompts, isLoading } = useQuery({
    queryKey: ['quickstart-prompts'],
    queryFn: getQuickstartPrompts,
  })

  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div className="flex gap-1 border-b border-slate-200 px-2 pt-2 dark:border-slate-800">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`rounded-t-lg px-3 py-1.5 text-xs font-semibold transition ${
              tab === t.id
                ? 'bg-slate-100 text-slate-900 dark:bg-slate-800 dark:text-slate-100'
                : 'text-slate-500 hover:text-slate-700 dark:text-slate-400'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="p-4">
        {isLoading || !prompts ? (
          <p className="text-sm text-slate-400">Loading quick-start prompts…</p>
        ) : (
          <>
            {tab === 'prompt' && (
              <div className="grid gap-2 sm:grid-cols-3">
                <StepButton
                  label="① Register download_mace_model"
                  hint="Sends full function source as part of the prompt"
                  colorClass="border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100 dark:border-blue-900 dark:bg-blue-950/40 dark:text-blue-300"
                  onClick={() => onSelectPrompt(prompts.t1a)}
                />
                <StepButton
                  label="② Register smiles_to_xyz + packmol_build_system"
                  hint="Sends both function sources inline"
                  colorClass="border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100 dark:border-blue-900 dark:bg-blue-950/40 dark:text-blue-300"
                  onClick={() => onSelectPrompt(prompts.t1b)}
                />
                <StepButton
                  label="③ Create mace_md_specialist"
                  hint="Natural-language request — no code pasted"
                  colorClass="border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100 dark:border-blue-900 dark:bg-blue-950/40 dark:text-blue-300"
                  onClick={() => onSelectPrompt(prompts.t1c)}
                />
              </div>
            )}

            {tab === 'script' && (
              <div className="grid gap-4 sm:grid-cols-2">
                <FileUploadZone onUploaded={onSelectPrompt} />
                <div>
                  <StepButton
                    label="④ Register run_nvt_md from ASE_NVT_PBC.py"
                    hint="Assigns run_nvt_md to mace_md_specialist"
                    colorClass="border-green-200 bg-green-50 text-green-700 hover:bg-green-100 dark:border-green-900 dark:bg-green-950/40 dark:text-green-300"
                    onClick={() => onSelectPrompt(prompts.t3a)}
                  />
                </div>
              </div>
            )}

            {tab === 'llm' && (
              <div className="grid gap-4 sm:grid-cols-3">
                <StepButton
                  label="⑤ Write & register plot_nvt_trajectory"
                  hint="LLM writes, registers, and assigns the tool"
                  colorClass="border-purple-200 bg-purple-50 text-purple-700 hover:bg-purple-100 dark:border-purple-900 dark:bg-purple-950/40 dark:text-purple-300"
                  onClick={() => onSelectPrompt(prompts.t4a)}
                />
                <pre className="col-span-2 max-h-40 overflow-auto rounded-lg bg-slate-50 p-2 font-mono text-xs text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                  {prompts.t4a}
                </pre>
              </div>
            )}

            {tab === 'run' && (
              <div className="grid gap-2 sm:grid-cols-3">
                <StepButton
                  label="T2 — Build NaCl + Water Box"
                  hint="1 NaCl pair + 267 H₂O · 20 Å periodic box"
                  colorClass="border-orange-200 bg-orange-50 text-orange-700 hover:bg-orange-100 dark:border-orange-900 dark:bg-orange-950/40 dark:text-orange-300"
                  onClick={() => onSelectPrompt(prompts.t2)}
                />
                <StepButton
                  label="T3 — Run NVT MD (10 steps, 300 K)"
                  hint="MACE-MP-0b3 · ASE · nacl_water_box.xyz"
                  colorClass="border-orange-200 bg-orange-50 text-orange-700 hover:bg-orange-100 dark:border-orange-900 dark:bg-orange-950/40 dark:text-orange-300"
                  onClick={() => onSelectPrompt(prompts.t3b)}
                />
                <StepButton
                  label="T4 — Plot NVT Trajectory"
                  hint="Energy & temperature from nvt_nacl_water.traj"
                  colorClass="border-orange-200 bg-orange-50 text-orange-700 hover:bg-orange-100 dark:border-orange-900 dark:bg-orange-950/40 dark:text-orange-300"
                  onClick={() => onSelectPrompt(prompts.t4b)}
                />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
