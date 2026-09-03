import { useQuery } from '@tanstack/react-query'
import { Bot, ClipboardList, FileCode, Play } from 'lucide-react'
import { useState } from 'react'
import { getQuickstartPrompts } from '../lib/api'
import { FileUploadZone } from './FileUploadZone'

interface QuickStartPanelProps {
  onSelectPrompt: (prompt: string) => void
}

type TabId = 'prompt' | 'script' | 'llm' | 'run'

const TABS: { id: TabId; label: string; icon: typeof ClipboardList }[] = [
  { id: 'prompt', label: 'From Prompt', icon: ClipboardList },
  { id: 'script', label: 'From Script', icon: FileCode },
  { id: 'llm', label: 'From LLM', icon: Bot },
  { id: 'run', label: 'Run Simulations', icon: Play },
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
      className={`w-full rounded-xl border px-3 py-2 text-left text-sm font-semibold transition ${colorClass}`}
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
    <div className="rounded-2xl border border-line bg-surface shadow-sm">
      <div className="flex gap-1 border-b border-line px-2 pt-2">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`flex items-center gap-1.5 rounded-t-lg px-3 py-1.5 text-xs font-semibold transition ${
              tab === id
                ? 'bg-brand-50 text-brand-800'
                : 'text-ink-muted hover:text-ink'
            }`}
          >
            <Icon className="h-3.5 w-3.5" />
            {label}
          </button>
        ))}
      </div>

      <div className="p-4">
        {isLoading || !prompts ? (
          <p className="text-sm text-ink-faint">Loading quick-start prompts…</p>
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
                    colorClass="border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300"
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
                  colorClass="border-violet-200 bg-violet-50 text-violet-700 hover:bg-violet-100 dark:border-violet-900 dark:bg-violet-950/40 dark:text-violet-300"
                  onClick={() => onSelectPrompt(prompts.t4a)}
                />
                <pre className="col-span-2 max-h-40 overflow-auto rounded-xl bg-surface-hover p-2 font-mono text-xs text-ink-muted">
                  {prompts.t4a}
                </pre>
              </div>
            )}

            {tab === 'run' && (
              <div className="grid gap-2 sm:grid-cols-3">
                <StepButton
                  label="T2 — Build NaCl + Water Box"
                  hint="1 NaCl pair + 267 H₂O · 20 Å periodic box"
                  colorClass="border-amber-200 bg-amber-50 text-amber-700 hover:bg-amber-100 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300"
                  onClick={() => onSelectPrompt(prompts.t2)}
                />
                <StepButton
                  label="T3 — Run NVT MD (10 steps, 300 K)"
                  hint="MACE-MP-0b3 · ASE · nacl_water_box.xyz"
                  colorClass="border-amber-200 bg-amber-50 text-amber-700 hover:bg-amber-100 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300"
                  onClick={() => onSelectPrompt(prompts.t3b)}
                />
                <StepButton
                  label="T4 — Plot NVT Trajectory"
                  hint="Energy & temperature from nvt_nacl_water.traj"
                  colorClass="border-amber-200 bg-amber-50 text-amber-700 hover:bg-amber-100 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300"
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
