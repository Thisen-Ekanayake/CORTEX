import { useState } from 'react'
import { motion } from 'framer-motion'
import { Card } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import './pages.css'

export function Settings() {
  const [darkMode, setDarkMode] = useState(true)
  const [compactSidebar, setCompactSidebar] = useState(false)
  const [highContrast, setHighContrast] = useState(false)
  const [animations, setAnimations] = useState(true)

  return (
    <motion.div
      className="page page--settings"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <h2 className="page__title">Settings</h2>
      <p className="page__subtitle">
        Dark mode toggle, layout customizations, and preferences.
      </p>

      <div className="settings-grid">
        <Card>
          <h3 className="card__heading">Appearance</h3>
          <div className="settings-group">
            <label className="toggle">
              <span
                className="toggle__track"
                data-checked={darkMode}
                onClick={() => setDarkMode((v) => !v)}
                role="switch"
                aria-checked={darkMode}
                tabIndex={0}
                onKeyDown={(e) => e.key === ' ' && setDarkMode((v) => !v)}
              >
                <span className="toggle__thumb" />
              </span>
              <span className="toggle__label">Dark mode</span>
            </label>
            <label className="toggle">
              <span
                className="toggle__track"
                data-checked={highContrast}
                onClick={() => setHighContrast((v) => !v)}
                role="switch"
                aria-checked={highContrast}
                tabIndex={0}
                onKeyDown={(e) => e.key === ' ' && setHighContrast((v) => !v)}
              >
                <span className="toggle__thumb" />
              </span>
              <span className="toggle__label">High contrast</span>
            </label>
          </div>
        </Card>

        <Card>
          <h3 className="card__heading">Layout</h3>
          <div className="settings-group">
            <label className="toggle">
              <span
                className="toggle__track"
                data-checked={compactSidebar}
                onClick={() => setCompactSidebar((v) => !v)}
                role="switch"
                aria-checked={compactSidebar}
                tabIndex={0}
                onKeyDown={(e) => e.key === ' ' && setCompactSidebar((v) => !v)}
              >
                <span className="toggle__thumb" />
              </span>
              <span className="toggle__label">Compact sidebar</span>
            </label>
          </div>
        </Card>

        <Card>
          <h3 className="card__heading">Accessibility</h3>
          <div className="settings-group">
            <label className="toggle">
              <span
                className="toggle__track"
                data-checked={animations}
                onClick={() => setAnimations((v) => !v)}
                role="switch"
                aria-checked={animations}
                tabIndex={0}
                onKeyDown={(e) => e.key === ' ' && setAnimations((v) => !v)}
              >
                <span className="toggle__thumb" />
              </span>
              <span className="toggle__label">Animations</span>
            </label>
          </div>
        </Card>

        <Card>
          <h3 className="card__heading">Profile</h3>
          <p className="settings-desc">Manage your profile and preferences.</p>
          <Button variant="secondary">Edit profile</Button>
        </Card>
      </div>
    </motion.div>
  )
}
