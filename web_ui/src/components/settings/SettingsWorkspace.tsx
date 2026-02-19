

export function SettingsWorkspace() {
    return (
        <div className="space-y-8">
            {/* Team Members */}
            <section className="glass-panel rounded-3xl p-6 md:p-8 space-y-6 border border-white/5">
                <div className="flex items-center justify-between pb-4 border-b border-white/5">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-orange-500/10 text-orange-500">
                            <span className="text-lg">👥</span>
                        </div>
                        <div>
                            <h3 className="text-base font-semibold text-text-primary">Team Members</h3>
                            <p className="text-xs text-text-tertiary">Manage access to your workspace</p>
                        </div>
                    </div>
                    <button className="px-4 py-2 rounded-xl bg-accent-primary text-white text-sm font-medium shadow-glow hover:bg-accent-primary/90 transition-colors">
                        Invite Member
                    </button>
                </div>

                <div className="space-y-4">
                    {[
                        { name: 'Thisen Ekanayake', email: 'thisen@cortex.ai', role: 'Owner', avatar: 'TE', color: 'from-accent-primary to-purple-600' },
                        { name: 'Sarah Connor', email: 'sarah@cortex.ai', role: 'Admin', avatar: 'SC', color: 'from-blue-500 to-cyan-500' },
                        { name: 'John Doe', email: 'john@cortex.ai', role: 'Viewer', avatar: 'JD', color: 'from-emerald-500 to-teal-500' },
                    ].map((member, i) => (
                        <div key={i} className="flex items-center justify-between group">
                            <div className="flex items-center gap-4">
                                <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${member.color} flex items-center justify-center text-white font-bold text-xs shadow-md`}>
                                    {member.avatar}
                                </div>
                                <div>
                                    <p className="text-sm font-medium text-text-primary">{member.name}</p>
                                    <p className="text-xs text-text-tertiary">{member.email}</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-4">
                                <span className="text-xs font-medium text-text-secondary bg-white/5 px-2 py-1 rounded-md border border-white/5">
                                    {member.role}
                                </span>
                                <button className="p-2 text-text-tertiary hover:text-text-primary hover:bg-white/5 rounded-lg transition-colors">
                                    •••
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            </section>

            {/* Roles & Permissions */}
            <section className="glass-panel rounded-3xl p-6 space-y-6 border border-white/5">
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-lg bg-pink-500/10 text-pink-500">
                        <span className="text-lg">🔑</span>
                    </div>
                    <h3 className="text-base font-semibold text-text-primary">Default Permissions</h3>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 rounded-xl bg-white/5 border border-white/5 hover:border-white/10 transition-colors cursor-pointer">
                        <div className="flex justify-between mb-2">
                            <span className="text-sm font-medium text-text-primary">Member Role</span>
                            <input type="radio" name="default_role" defaultChecked className="accent-accent-primary" />
                        </div>
                        <p className="text-xs text-text-tertiary">Can create new chats, view shared documents, and edit profile.</p>
                    </div>
                    <div className="p-4 rounded-xl bg-white/5 border border-white/5 hover:border-white/10 transition-colors cursor-pointer">
                        <div className="flex justify-between mb-2">
                            <span className="text-sm font-medium text-text-primary">Viewer Role</span>
                            <input type="radio" name="default_role" className="accent-accent-primary" />
                        </div>
                        <p className="text-xs text-text-tertiary">Read-only access to public projects and documents.</p>
                    </div>
                </div>
            </section>
        </div>
    )
}
