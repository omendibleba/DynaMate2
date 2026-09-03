import { MessageSquare, Sparkles, Workflow } from 'lucide-react'
import logo from '../assets/dynamate2-logo.png'

export type TabId = 'chat' | 'quickstart' | 'graph'

const TABS: { id: TabId; label: string; icon: typeof MessageSquare }[] = [
  { id: 'chat', label: 'Chat', icon: MessageSquare },
  { id: 'quickstart', label: 'Quick Start', icon: Sparkles },
  { id: 'graph', label: 'Agent Graph', icon: Workflow },
]

interface NavBarProps {
  tab: TabId
  onTabChange: (tab: TabId) => void
}

export function NavBar({ tab, onTabChange }: NavBarProps) {
  return (
    <header className="flex items-center justify-between gap-4 border-b border-line px-1 py-2">
      <div className="flex items-center gap-2.5">
        <img src={logo} alt="DynaMate2" className="h-8 w-8 rounded-md" />
        <span className="text-base font-semibold tracking-tight text-ink">DynaMate2</span>
      </div>

      <nav className="flex items-center gap-1">
        {TABS.map(({ id, label, icon: Icon }) => {
          const active = tab === id
          return (
            <button
              key={id}
              onClick={() => onTabChange(id)}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition outline-none focus-visible:ring-2 focus-visible:ring-accent/50 ${
                active
                  ? 'bg-surface-hover text-ink border-b-2 border-accent'
                  : 'text-ink-muted hover:bg-surface-hover hover:text-ink border-b-2 border-transparent'
              }`}
            >
              <Icon className="h-4 w-4" />
              {label}
            </button>
          )
        })}
      </nav>
    </header>
  )
}
