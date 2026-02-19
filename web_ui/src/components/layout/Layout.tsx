import { useState } from 'react'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'
import { ContextPanel } from '../chat/ContextPanel'
import { Outlet } from 'react-router-dom'

export function Layout() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  // Only show context panel on Chat page for now, or always?
  // Let's show it always if screen is large enough, as per "Professional UI"
  const showContextPanel = true

  return (
    <div className="flex h-screen overflow-hidden bg-bg-base text-text-primary font-sans">
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((c) => !c)}
      />

      <div className="flex-1 flex flex-col min-w-0 transition-all duration-300">
        <TopBar />

        <main className="flex-1 flex overflow-hidden relative">
          <div className="flex-1 overflow-y-auto overflow-x-hidden relative scroll-smooth">
            <Outlet />
          </div>

          {showContextPanel && <ContextPanel />}
        </main>
      </div>
    </div>
  )
}
