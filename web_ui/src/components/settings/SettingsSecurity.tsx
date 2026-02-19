import { useState } from 'react'
import { ToggleRow } from './ToggleRow'

export function SettingsSecurity() {
    const [twoFactor, setTwoFactor] = useState(false)
    const [sessionTimeout, setSessionTimeout] = useState(true)

    return (
        <div className="space-y-8">
            {/* Password Section */}
            <section className="glass-panel rounded-3xl p-6 md:p-8 space-y-6 border border-white/5">
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg bg-green-500/10 text-green-500">
                        <span className="text-lg">🔒</span>
                    </div>
                    <h3 className="text-base font-semibold text-text-primary">Password & Authentication</h3>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                        <label className="text-xs font-semibold uppercase tracking-wider text-text-tertiary">Current Password</label>
                        <input
                            type="password"
                            placeholder="••••••••••••"
                            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-text-primary focus:outline-none focus:border-accent-primary/50 transition-colors"
                        />
                    </div>
                    {/* Placeholder for layout balance */}
                    <div className="hidden md:block"></div>

                    <div className="space-y-2">
                        <label className="text-xs font-semibold uppercase tracking-wider text-text-tertiary">New Password</label>
                        <input
                            type="password"
                            placeholder="Enter new password"
                            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-text-primary focus:outline-none focus:border-accent-primary/50 transition-colors"
                        />
                    </div>
                    <div className="space-y-2">
                        <label className="text-xs font-semibold uppercase tracking-wider text-text-tertiary">Confirm New Password</label>
                        <input
                            type="password"
                            placeholder="Confirm new password"
                            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-text-primary focus:outline-none focus:border-accent-primary/50 transition-colors"
                        />
                    </div>
                </div>

                <div className="pt-4 border-t border-white/5">
                    <button className="px-5 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 text-sm font-medium transition-all text-text-primary border border-white/10">
                        Update Password
                    </button>
                </div>
            </section>

            {/* 2FA and Sessions */}
            <section className="glass-panel rounded-3xl p-6 space-y-6 border border-white/5">
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg bg-blue-500/10 text-blue-500">
                        <span className="text-lg">🛡️</span>
                    </div>
                    <h3 className="text-base font-semibold text-text-primary">Security Preferences</h3>
                </div>

                <div className="space-y-6">
                    <ToggleRow
                        label="Two-Factor Authentication (2FA)"
                        description="Add an extra layer of security to your account"
                        checked={twoFactor}
                        onChange={setTwoFactor}
                    />
                    <ToggleRow
                        label="Automatic Session Timeout"
                        description="Log out after 30 minutes of inactivity"
                        checked={sessionTimeout}
                        onChange={setSessionTimeout}
                    />
                </div>
            </section>

            {/* Active Sessions */}
            <section className="glass-panel rounded-3xl p-6 space-y-6 border border-white/5">
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg bg-purple-500/10 text-purple-500">
                        <span className="text-lg">💻</span>
                    </div>
                    <h3 className="text-base font-semibold text-text-primary">Active Sessions</h3>
                </div>

                <div className="space-y-3">
                    {[
                        { device: 'Chrome on MacBook Pro', location: 'Colombo, Sri Lanka', active: true, ip: '192.168.1.1' },
                        { device: 'Safari on iPhone 15', location: 'Colombo, Sri Lanka', active: false, ip: '192.168.1.45' },
                    ].map((session, i) => (
                        <div key={i} className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5">
                            <div className="flex items-center gap-3">
                                <div className="w-2 h-2 rounded-full bg-accent-success/50" />
                                <div>
                                    <p className="text-sm font-medium text-text-primary">{session.device}</p>
                                    <p className="text-xs text-text-tertiary">{session.location} • {session.ip}</p>
                                </div>
                            </div>
                            {session.active ? (
                                <span className="text-xs font-medium text-accent-success bg-accent-success/10 px-2 py-1 rounded-md">Current</span>
                            ) : (
                                <button className="text-xs text-accent-error hover:text-red-400 transition-colors">Revoke</button>
                            )}
                        </div>
                    ))}
                </div>
            </section>
        </div>
    )
}
