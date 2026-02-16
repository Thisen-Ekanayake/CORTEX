import { useState } from 'react'
import { ModelSelector } from '../chat/ModelSelector'

export function TopBar() {
  const [showUserMenu, setShowUserMenu] = useState(false)

  return (
    <header className="h-[var(--top-bar-height)] flex items-center justify-between px-4 border-b border-border-subtle bg-surface-glass/50 backdrop-blur-md sticky top-0 z-40">
      <div className="flex items-center gap-4">
        {/* Model Selector */}
        <ModelSelector />

        {/* Breadcrumb / Context */}
        <div className="h-4 w-px bg-border-subtle" />
        <span className="text-sm text-text-secondary truncate max-w-[200px] lg:max-w-md">
          New Chat
        </span>
      </div>

      <div className="flex items-center gap-2">
        {/* Search Trigger */}
        <button className="p-2 text-text-secondary hover:text-text-primary hover:bg-white/5 rounded-md transition-colors">
          <span className="text-lg">⌕</span>
        </button>

        {/* Share Button */}
        <button className="hidden sm:flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-text-primary bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg transition-colors">
          <span>Share</span>
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
