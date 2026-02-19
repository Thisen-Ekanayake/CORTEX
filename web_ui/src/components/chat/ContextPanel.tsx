import { motion, AnimatePresence } from 'framer-motion'

export function ContextPanel() {
    return (
        <aside className="w-80 border-l border-border-subtle bg-surface-glass/30 flex flex-col hidden lg:flex">
            {/* Tabs */}
            <div className="flex items-center px-4 py-3 border-b border-border-subtle gap-4">
                <div className="text-sm font-medium pb-2 border-b-2 border-accent-primary text-text-primary">
                    Sources
                </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                <AnimatePresence mode="wait">
                    <motion.div
                        key="sources"
                        initial={{ opacity: 0, x: 10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -10 }}
                        className="space-y-4"
                    >
                        <div className="p-3 rounded-lg border border-border-subtle bg-bg-elevated/50">
                            <div className="flex items-center gap-2 mb-2">
                                <span className="text-accent-success text-xs">●</span>
                                <span className="text-xs font-medium text-text-secondary uppercase">Source</span>
                            </div>
                            <h4 className="text-sm font-medium text-text-primary mb-1">
                                Q3 Financial Report.pdf
                            </h4>
                            <p className="text-xs text-text-secondary line-clamp-2">
                                Revenue increased by 15% YoY driven by strong enterprise adoption...
                            </p>
                        </div>

                        <div className="p-3 rounded-lg border border-border-subtle bg-bg-elevated/50">
                            <div className="flex items-center gap-2 mb-2">
                                <span className="text-accent-success text-xs">●</span>
                                <span className="text-xs font-medium text-text-secondary uppercase">Source</span>
                            </div>
                            <h4 className="text-sm font-medium text-text-primary mb-1">
                                Competitor Analysis 2025
                            </h4>
                            <p className="text-xs text-text-secondary line-clamp-2">
                                Key competitors are shifting towards hybrid cloud solutions...
                            </p>
                        </div>
                    </motion.div>
                </AnimatePresence>
            </div>
        </aside>
    )
}
