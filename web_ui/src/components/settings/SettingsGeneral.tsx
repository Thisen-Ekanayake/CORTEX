import { useState } from 'react'
import { ToggleRow } from './ToggleRow'

export function SettingsGeneral() {
    const [darkMode, setDarkMode] = useState(true)
    const [highContrast, setHighContrast] = useState(false)
    const [animations, setAnimations] = useState(true)
    const [compactSidebar, setCompactSidebar] = useState(false)
    const [emailNotifications, setEmailNotifications] = useState(true)

    return (
        <div className="space-y-8">
            {/* Section: Profile */}
            <section className="glass-panel rounded-3xl p-6 md:p-8 space-y-8 shadow-premium-blue/10 border border-white/5">
                <div className="flex items-center gap-6 pb-6 border-b border-white/5">
                    <div className="w-20 h-20 rounded-2xl bg-gradient-to-tr from-accent-primary to-purple-600 flex items-center justify-center text-white text-2xl font-bold shadow-glow ring-4 ring-white/5">
                        TE
                    </div>
                    <div className="flex-1">
                        <h3 className="text-xl font-semibold text-text-primary leading-none mb-2">Thisen Ekanayake</h3>
                        <p className="text-sm text-text-tertiary">Personal Professional Account</p>
                    </div>
                    <button className="px-5 py-2 rounded-xl bg-white/5 hover:bg-white/10 text-sm font-medium transition-all border border-white/10 hover:border-white/20">
                        Edit Profile
                    </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2 group">
                        <label className="text-xs font-semibold uppercase tracking-wider text-text-tertiary group-hover:text-text-secondary transition-colors">Workspace Name</label>
                        <input
                            type="text"
                            defaultValue="CORTEX Labs"
                            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-text-primary focus:outline-none focus:border-accent-primary/50 focus:bg-white/10 transition-all placeholder:text-text-tertiary/50"
                        />
                    </div>
                    <div className="space-y-2 group">
                        <label className="text-xs font-semibold uppercase tracking-wider text-text-tertiary group-hover:text-text-secondary transition-colors">Organization ID</label>
                        <div className="flex gap-2">
                            <input
                                type="text"
                                readOnly
                                value="ctx_8892_prod"
                                className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-text-tertiary cursor-not-allowed select-none"
                            />
                            <button className="p-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-colors text-text-tertiary hover:text-text-primary" title="Copy ID">
                                ❐
                            </button>
                        </div>
                    </div>
                </div>
            </section>

            {/* Section: Customization */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <section className="glass-panel rounded-3xl p-6 space-y-6 border border-white/5">
                    <div className="flex items-center gap-3 mb-2">
                        <div className="p-2 rounded-lg bg-accent-primary/10 text-accent-primary">
                            <span className="text-lg">✨</span>
                        </div>
                        <h3 className="text-base font-semibold text-text-primary">Appearance</h3>
                    </div>

                    <div className="space-y-4">
                        <ToggleRow
                            label="Dark mode"
                            description="Optimize for low light environments"
                            checked={darkMode}
                            onChange={setDarkMode}
                        />
                        <ToggleRow
                            label="High contrast"
                            description="Increase visibility for UI elements"
                            checked={highContrast}
                            onChange={setHighContrast}
                        />
                        <ToggleRow
                            label="Reduced motion"
                            description="Disable complex UI animations"
                            checked={!animations}
                            onChange={(v: boolean) => setAnimations(!v)}
                        />
                    </div>
                </section>

                <section className="glass-panel rounded-3xl p-6 space-y-6 border border-white/5">
                    <div className="flex items-center gap-3 mb-2">
                        <div className="p-2 rounded-lg bg-yellow-500/10 text-yellow-500">
                            <span className="text-lg">⚡</span>
                        </div>
                        <h3 className="text-base font-semibold text-text-primary">Workflow</h3>
                    </div>

                    <div className="space-y-4">
                        <ToggleRow
                            label="Compact sidebar"
                            description="Minimize sidebar to icons only"
                            checked={compactSidebar}
                            onChange={setCompactSidebar}
                        />
                        <ToggleRow
                            label="Email alerts"
                            description="Receive updates on background tasks"
                            checked={emailNotifications}
                            onChange={setEmailNotifications}
                        />
                    </div>
                </section>
            </div>

            {/* Action Footer */}
            <div className="flex justify-end gap-4 pt-4">
                <button className="px-6 py-2.5 rounded-xl text-sm font-medium text-text-secondary hover:text-text-primary transition-colors hover:bg-white/5">
                    Discard Changes
                </button>
                <button className="px-8 py-2.5 rounded-xl bg-accent-primary text-white text-sm font-semibold shadow-premium-blue hover:shadow-premium-blue/80 hover:translate-y-[-1px] active:translate-y-[0px] transition-all">
                    Save Settings
                </button>
            </div>
        </div>
    )
}
