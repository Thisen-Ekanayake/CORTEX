import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { SettingsGeneral } from '../components/settings/SettingsGeneral'
import { SettingsSecurity } from '../components/settings/SettingsSecurity'
import { SettingsWorkspace } from '../components/settings/SettingsWorkspace'
import { SettingsNotifications } from '../components/settings/SettingsNotifications'
import { SettingsData } from '../components/settings/SettingsData'

type SettingsTab = 'General' | 'Security' | 'Workspace' | 'Notifications' | 'Data'

export function Settings() {
  const [activeTab, setActiveTab] = useState<SettingsTab>('General')

  const renderContent = () => {
    switch (activeTab) {
      case 'General': return <SettingsGeneral />
      case 'Security': return <SettingsSecurity />
      case 'Workspace': return <SettingsWorkspace />
      case 'Notifications': return <SettingsNotifications />
      case 'Data': return <SettingsData />
      default: return <SettingsGeneral />
    }
  }

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
            {(['General', 'Security', 'Workspace', 'Notifications', 'Data'] as SettingsTab[]).map((item) => (
              <button
                key={item}
                onClick={() => setActiveTab(item)}
                className={`w-full text-left px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${activeTab === item
                  ? 'bg-accent-primary/10 text-accent-primary border border-accent-primary/20 shadow-sm'
                  : 'text-text-secondary hover:text-text-primary hover:bg-white/5 border border-transparent'
                  }`}
              >
                {item}
              </button>
            ))}
          </nav>
        </div>

        {/* Content Section */}
        <div className="lg:col-span-9">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={{ duration: 0.2 }}
            >
              {renderContent()}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  )
}


