import { useState } from 'react'
import { motion } from 'framer-motion'

export function Settings() {
  const [darkMode, setDarkMode] = useState(true)
  const [compactSidebar, setCompactSidebar] = useState(false)
  const [highContrast, setHighContrast] = useState(false)
  const [animations, setAnimations] = useState(true)
  const [emailNotifications, setEmailNotifications] = useState(true)

  return (
    <motion.div
      className="max-w-6xl mx-auto p-4 md:p-10 space-y-12"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.19, 1, 0.22, 1] }}
    >
      {/* Header Section */}
      <div className="flex flex-col gap-2">
        <h2 className="text-4xl font-bold bg-gradient-to-r from-text-primary to-text-secondary bg-clip-text text-transparent">
          Settings
        </h2>
        <p className="text-text-secondary text-[15px]">
          Configure your workspace, appearance, and personal preferences.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
        {/* Navigation - Sticky Left */}
        <div className="lg:col-span-3">
          <nav className="sticky top-24 space-y-1">
            {['General', 'Security', 'Workspace', 'Notifications', 'Data'].map((item, i) => (
              <button
                key={item}
                className={`w-full text-left px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${i === 0
                  ? 'bg-accent-primary/10 text-accent-primary border border-accent-primary/20'
                  : 'text-text-secondary hover:text-text-primary hover:bg-white/5'
                  }`}
              >
                {item}
              </button>
            ))}
          </nav>
        </div>

        {/* Content Section */}
        <div className="lg:col-span-9 space-y-8">

          {/* Section: Profile */}
          <section className="glass-panel rounded-3xl p-6 md:p-8 space-y-8 shadow-premium-glass">
            <div className="flex items-center gap-6 pb-6 border-b border-white/5">
              <div className="w-20 h-20 rounded-2xl bg-gradient-to-tr from-accent-primary to-purple-600 flex items-center justify-center text-white text-2xl font-bold shadow-glow">
                TE
              </div>
              <div className="flex-1">
                <h3 className="text-xl font-semibold text-text-primary leading-none mb-2">Thisen Ekanayake</h3>
                <p className="text-sm text-text-tertiary">Personal Professional Account</p>
              </div>
              <button className="px-5 py-2 rounded-xl bg-white/5 hover:bg-white/10 text-sm font-medium transition-all border border-white/10">
                Edit Profile
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-wider text-text-tertiary">Workspace Name</label>
                <input
                  type="text"
                  defaultValue="CORTEX Labs"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-text-primary focus:outline-none focus:border-accent-primary/50 transition-colors"
                />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-wider text-text-tertiary">Organization ID</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    readOnly
                    value="ctx_8892_prod"
                    className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-text-tertiary cursor-not-allowed"
                  />
                  <button className="p-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-colors">
                    ❐
                  </button>
                </div>
              </div>
            </div>
          </section>

          {/* Section: Customization */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <section className="glass-panel rounded-3xl p-6 space-y-6">
              <div className="flex items-center gap-3 mb-2">
                <span className="text-lg">✨</span>
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
                  onChange={(v) => setAnimations(!v)}
                />
              </div>
            </section>

            <section className="glass-panel rounded-3xl p-6 space-y-6">
              <div className="flex items-center gap-3 mb-2">
                <span className="text-lg">⚡</span>
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
          <div className="flex justify-end gap-4 pt-10 border-t border-white/5">
            <button className="px-6 py-2.5 rounded-xl text-sm font-medium text-text-secondary hover:text-text-primary transition-colors">
              Discard Changes
            </button>
            <button className="px-8 py-2.5 rounded-xl bg-accent-primary text-white text-sm font-semibold shadow-glow hover:translate-y-[-1px] active:translate-y-[0px] transition-all">
              Save Settings
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function ToggleRow({ label, description, checked, onChange }: {
  label: string,
  description: string,
  checked: boolean,
  onChange: (v: boolean) => void
}) {
  return (
    <div className="flex items-center justify-between group">
      <div className="flex flex-col">
        <span className="text-sm font-medium text-text-primary group-hover:text-accent-primary transition-colors">{label}</span>
        <span className="text-xs text-text-tertiary">{description}</span>
      </div>
      <button
        onClick={() => onChange(!checked)}
        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-all duration-300 outline-none ${checked ? 'bg-accent-primary' : 'bg-white/10'
          }`}
      >
        <span
          className={`inline-block h-4 w-4 transform rounded-full bg-white transition-all duration-300 ${checked ? 'translate-x-6' : 'translate-x-1'
            }`}
        />
      </button>
    </div>
  )
}

