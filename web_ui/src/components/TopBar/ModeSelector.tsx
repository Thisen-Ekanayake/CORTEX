import { useState, useRef, useEffect } from 'react'
import type { IntelligenceMode } from '../../types'

const modes: IntelligenceMode[] = [
  'Ask',
  'Analyze',
  'Compare',
  'Summarize',
  'Draft',
  'Investigate',
]

type ModeSelectorProps = {
  selected: IntelligenceMode
  onSelect: (mode: IntelligenceMode) => void
}

export function ModeSelector({ selected, onSelect }: ModeSelectorProps) {
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
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span>{selected}</span>
        <span className="text-text-tertiary">&#9662;</span>
      </button>
      {open && (
        <ul
          role="listbox"
          className="absolute left-0 top-full mt-2 min-w-[160px] bg-surface-panel rounded-2xl shadow-panel z-50 py-2"
        >
          {modes.map((m) => (
            <li key={m} role="option" aria-selected={selected === m}>
              <button
                type="button"
                onClick={() => {
                  onSelect(m)
                  setOpen(false)
                }}
                className={`w-full text-left px-4 py-2.5 text-sm rounded-lg mx-1 transition-colors duration-200 ${
                  selected === m
                    ? 'text-blue-400 bg-surface-hover'
                    : 'text-text-primary hover:bg-surface-hover'
                }`}
              >
                {m}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
