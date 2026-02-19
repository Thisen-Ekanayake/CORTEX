

export function SettingsData() {
    return (
        <div className="space-y-8">
            {/* API Access */}
            <section className="glass-panel rounded-3xl p-6 md:p-8 space-y-6 border border-white/5">
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-500">
                        <span className="text-lg">🤖</span>
                    </div>
                    <h3 className="text-base font-semibold text-text-primary">API Access</h3>
                </div>

                <div className="space-y-4">
                    <div className="p-4 rounded-xl bg-bg-base/50 border border-white/10 flex items-center justify-between">
                        <div>
                            <label className="text-xs font-semibold uppercase tracking-wider text-text-tertiary">Production Key</label>
                            <div className="flex items-center gap-2 mt-1">
                                <code className="text-sm font-mono text-accent-primary">sk_live_51M...9A2z</code>
                                <span className="text-xs text-text-tertiary">• Created 2 months ago</span>
                            </div>
                        </div>
                        <div className="flex gap-2">
                            <button className="px-3 py-1.5 text-xs font-medium text-text-secondary bg-white/5 hover:bg-white/10 rounded-lg transition-colors">Rotate</button>
                            <button className="px-3 py-1.5 text-xs font-medium text-accent-error bg-accent-error/10 hover:bg-accent-error/20 rounded-lg transition-colors">Revoke</button>
                        </div>
                    </div>
                    <button className="w-full py-2.5 rounded-xl border border-dashed border-white/20 text-text-secondary text-sm font-medium hover:bg-white/5 hover:border-white/30 transition-all">
                        + Generate New API Key
                    </button>
                </div>
            </section>

            {/* Data Export */}
            <section className="glass-panel rounded-3xl p-6 space-y-6 border border-white/5">
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg bg-violet-500/10 text-violet-500">
                        <span className="text-lg">📥</span>
                    </div>
                    <h3 className="text-base font-semibold text-text-primary">Data Export</h3>
                </div>

                <p className="text-sm text-text-secondary">Download a copy of your personal data, including chat history, uploaded documents, and usage logs.</p>

                <div className="flex gap-3">
                    <button className="px-4 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-sm font-medium transition-colors">
                        Export Options
                    </button>
                    <button className="px-4 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-sm font-medium transition-colors">
                        Download Latest Archive (ZIP)
                    </button>
                </div>
            </section>

            {/* Danger Zone */}
            <section className="glass-panel rounded-3xl p-6 space-y-6 border border-accent-error/20 bg-accent-error/[0.02]">
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg bg-accent-error/10 text-accent-error">
                        <span className="text-lg">⚠️</span>
                    </div>
                    <h3 className="text-base font-semibold text-accent-error">Danger Zone</h3>
                </div>

                <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 rounded-xl border border-accent-error/10 bg-bg-base/50">
                        <div>
                            <h4 className="text-sm font-medium text-text-primary mb-1">Delete Workspace</h4>
                            <p className="text-xs text-text-tertiary max-w-sm">Permanently delete your workspace and all associated data. This action cannot be undone.</p>
                        </div>
                        <button className="px-4 py-2 rounded-lg bg-accent-error text-white text-xs font-bold hover:bg-accent-error/90 transition-colors shadow-lg shadow-accent-error/20">
                            Delete Workspace
                        </button>
                    </div>
                </div>
            </section>
        </div>
    )
}
