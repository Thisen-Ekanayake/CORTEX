import { useState, useRef, useEffect } from 'react'
import { user } from '../../data/mockData'

type UserDropdownProps = {
  onLogout?: () => void
}

export function UserDropdown({ onLogout }: UserDropdownProps) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-4 py-2.5 text-sm text-text-primary bg-surface-hover rounded-xl hover:bg-surface-hover hover:text-text-primary transition-colors duration-200"
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <span className="text-text-secondary">{user.name}</span>
        <span className="text-text-tertiary">&#9662;</span>
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full mt-2 min-w-[200px] bg-surface-panel rounded-2xl shadow-panel z-50 py-2"
        >
          <div className="px-4 py-3 mb-2 mx-2 rounded-xl bg-surface-hover/50">
            <div className="text-sm font-medium text-text-primary">{user.name}</div>
            <div className="text-xs text-text-tertiary">{user.username}</div>
            <div className="text-xs text-text-secondary">{user.role}</div>
          </div>
          <button
            type="button"
            role="menuitem"
            className="w-full text-left px-4 py-2.5 text-sm text-text-primary hover:bg-surface-hover rounded-lg mx-2 transition-colors duration-200"
          >
            Settings
          </button>
          <button
            type="button"
            role="menuitem"
            className="w-full text-left px-4 py-2.5 text-sm text-text-primary hover:bg-surface-hover rounded-lg mx-2 transition-colors duration-200"
          >
            Help
          </button>
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              onLogout?.()
              setOpen(false)
            }}
            className="w-full text-left px-4 py-2.5 text-sm text-text-primary hover:bg-surface-hover rounded-lg mx-2 transition-colors duration-200"
          >
            Logout
          </button>
        </div>
      )}
    </div>
  )
}
