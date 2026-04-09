import { useState, useRef, useEffect } from 'react'
import { workspaces } from '../../data/mockData'

type WorkspaceSelectorProps = {
  selectedId: string
  onSelect: (id: string) => void
}

export function WorkspaceSelector({ selectedId, onSelect }: WorkspaceSelectorProps) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const selected = workspaces.find((w) => w.id === selectedId) ?? workspaces[0]

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
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span>{selected.label}</span>
        <span className="text-text-tertiary">&#9662;</span>
      </button>
      {open && (
        <ul
          role="listbox"
          className="absolute left-0 top-full mt-2 min-w-[140px] bg-surface-panel rounded-2xl shadow-panel z-50 py-2"
        >
          {workspaces.map((w) => (
            <li key={w.id} role="option" aria-selected={selectedId === w.id}>
              <button
                type="button"
                onClick={() => {
                  onSelect(w.id)
                  setOpen(false)
                }}
                className={`w-full text-left px-4 py-2.5 text-sm rounded-lg mx-1 transition-colors duration-200 ${
                  selectedId === w.id
                    ? 'text-blue-400 bg-surface-hover'
                    : 'text-text-primary hover:bg-surface-hover'
                }`}
              >
                {w.label}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
