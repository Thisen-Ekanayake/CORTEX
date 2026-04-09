import { useState } from 'react'
import { ToggleRow } from './ToggleRow'

export function SettingsNotifications() {
    const [emailDigest, setEmailDigest] = useState(true)
    const [pushMentions, setPushMentions] = useState(true)
    const [pushUpdates, setPushUpdates] = useState(false)
    const [slackIntegration, setSlackIntegration] = useState(true)

    return (
        <div className="space-y-8">
            {/* Email Notifications */}
            <section className="glass-panel rounded-3xl p-6 space-y-6 border border-white/5">
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-500">
                        <span className="text-lg">📧</span>
                    </div>
                    <h3 className="text-base font-semibold text-text-primary">Email Notifications</h3>
                </div>

                <div className="space-y-4">
                    <ToggleRow
                        label="Weekly Digest"
                        description="Summary of workspace activity and performance stats"
                        checked={emailDigest}
                        onChange={setEmailDigest}
                    />
                    <ToggleRow
                        label="Product Updates"
                        description="News about new features and improvements"
                        checked={true}
                        onChange={() => { }}
                    />
                </div>
            </section>

            {/* Push Notifications */}
            <section className="glass-panel rounded-3xl p-6 space-y-6 border border-white/5">
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg bg-rose-500/10 text-rose-500">
                        <span className="text-lg">🔔</span>
                    </div>
                    <h3 className="text-base font-semibold text-text-primary">Push Notifications</h3>
                </div>

                <div className="space-y-4">
                    <ToggleRow
                        label="Mentions & Replies"
                        description="When someone mentions you or replies to your comment"
                        checked={pushMentions}
                        onChange={setPushMentions}
                    />
                    <ToggleRow
                        label="System Updates"
                        description="Important system status and maintenance alerts"
                        checked={pushUpdates}
                        onChange={setPushUpdates}
                    />
                </div>
            </section>

            {/* Integrations */}
            <section className="glass-panel rounded-3xl p-6 space-y-6 border border-white/5">
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg bg-teal-500/10 text-teal-500">
                        <span className="text-lg">🔌</span>
                    </div>
                    <h3 className="text-base font-semibold text-text-primary">Integrations</h3>
                </div>

                <div className="space-y-4">
                    <div className="flex items-center justify-between py-2">
                        <div className="flex items-center gap-4">
                            <div className="w-10 h-10 rounded-lg bg-white flex items-center justify-center">
                                <span className="text-black font-bold text-lg">#</span>
                            </div>
                            <div>
                                <p className="text-sm font-medium text-text-primary">Slack</p>
                                <p className="text-xs text-text-tertiary">Send alerts to #general channel</p>
                            </div>
                        </div>
                        <ToggleRow
                            label=""
                            description=""
                            checked={slackIntegration}
                            onChange={setSlackIntegration}
                        />
                    </div>
                </div>
            </section>
        </div>
    )
}
