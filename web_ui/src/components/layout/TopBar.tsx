import { useState } from 'react'
import { useLocation } from 'react-router-dom'
import { ModelSelector } from '../chat/ModelSelector'

export function TopBar() {
  const [showUserMenu, setShowUserMenu] = useState(false)
  const location = useLocation()
  const isSettingsPage = location.pathname === '/settings'

  return (
    <header className="h-[var(--top-bar-height)] flex items-center justify-between px-4 border-b border-border-subtle bg-bg-base/80 backdrop-blur-xl sticky top-0 z-40">
      <div className="flex items-center gap-4">
        {/* Model Selector */}
        {!isSettingsPage && <ModelSelector />}
      </div>

      <div className="flex items-center gap-2">
        {/* Search Trigger */}
        <button className="p-2 text-text-secondary hover:text-text-primary hover:bg-white/5 rounded-md transition-colors">
          <span className="text-lg">⌕</span>
        </button>

        {/* User Profile */}
        <div className="relative">
          <button
            onClick={() => setShowUserMenu(!showUserMenu)}
            className="flex items-center justify-center w-8 h-8 rounded-full bg-gradient-to-tr from-accent-primary to-purple-500 text-white font-medium text-xs shadow-glow hover:opacity-90 transition-opacity"
          >
            TE
          </button>
        </div>
      </div>
    </header>
  )
}
