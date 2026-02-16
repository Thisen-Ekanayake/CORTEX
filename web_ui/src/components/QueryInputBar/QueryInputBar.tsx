import { useState } from 'react'

type QueryInputBarProps = {
  value: string
  onChange: (value: string) => void
  onRun: () => void
  disabled?: boolean
}

export function QueryInputBar({
  value,
  onChange,
  onRun,
  disabled = false,
}: QueryInputBarProps) {
  const [advanced, setAdvanced] = useState(false)

  return (
    <div className="flex items-center gap-4 p-4 bg-surface-panel rounded-2xl shadow-panel">
      <button
        type="button"
        className="flex-shrink-0 px-4 py-3 text-sm text-text-secondary bg-surface-hover rounded-xl hover:bg-surface-hover hover:text-text-primary transition-colors duration-200"
      >
        + Attach
      </button>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && onRun()}
        placeholder="Enter query..."
        className="flex-1 min-w-0 px-5 py-3.5 text-base text-text-primary bg-surface-hover rounded-xl placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-surface-panel transition-shadow duration-200"
        disabled={disabled}
      />
      <div className="flex-shrink-0 flex items-center gap-2">
        <button
          type="button"
          onClick={() => setAdvanced(!advanced)}
          className={`px-4 py-3 text-sm rounded-xl transition-colors duration-200 ${
            advanced
              ? 'bg-blue-500/20 text-blue-400'
              : 'text-text-secondary bg-surface-hover hover:bg-surface-hover hover:text-text-primary'
          }`}
        >
          Advanced
        </button>
        <button
          type="button"
          className="p-3 text-text-secondary bg-surface-hover rounded-xl hover:bg-surface-hover hover:text-text-primary transition-colors duration-200"
          aria-label="Microphone"
        >
          Mic
        </button>
        <button
          type="button"
          onClick={onRun}
          disabled={disabled || !value.trim()}
          className="px-5 py-3.5 text-sm font-medium text-white bg-blue-600 rounded-xl hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200"
        >
          Run
        </button>
      </div>
    </div>
  )
}
