import { useState } from 'react'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'
import './layout.css'

type LayoutProps = { children: React.ReactNode }

export function Layout({ children }: LayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  return (
    <div className="layout">
      <div className="layout__overlay" aria-hidden />
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((c) => !c)}
      />
      <div className="layout__main">
        <TopBar />
        <main id="main" className="layout__content" role="main">
          {children}
        </main>
      </div>
    </div>
  )
}
