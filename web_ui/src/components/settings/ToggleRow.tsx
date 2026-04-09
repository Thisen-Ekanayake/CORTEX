export function ToggleRow({ label, description, checked, onChange }: {
    label: string,
    description: string,
    checked: boolean,
    onChange: (v: boolean) => void
}) {
    return (
        <div className="flex items-center justify-between group py-1">
            <div className="flex flex-col">
                <span className="text-sm font-medium text-text-primary group-hover:text-accent-primary transition-colors">{label}</span>
                <span className="text-xs text-text-tertiary">{description}</span>
            </div>
            <button
                onClick={() => onChange(!checked)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-all duration-300 outline-none focus:ring-2 focus:ring-accent-primary/50 focus:ring-offset-2 focus:ring-offset-bg-base ${checked ? 'bg-accent-primary' : 'bg-white/10 group-hover:bg-white/20'
                    }`}
            >
                <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white shadow-sm transition-all duration-300 ${checked ? 'translate-x-6' : 'translate-x-1'
                        }`}
                />
            </button>
        </div>
    )
}
