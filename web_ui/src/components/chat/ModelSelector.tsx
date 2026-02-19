import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const MODELS = [
    { id: 'gpt-4', name: 'GPT-4', description: 'Most capable model for complex tasks' },
    { id: 'gpt-3.5', name: 'GPT-3.5', description: 'Fast and efficient for everyday tasks' },
    { id: 'local-mistral', name: 'Mistral 7B (Local)', description: 'Private, local execution' },
]

export function ModelSelector() {
    const [isOpen, setIsOpen] = useState(false)
    const [selectedModel, setSelectedModel] = useState(MODELS[0])

    return (
        <div className="relative z-50">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-white/5 transition-colors text-text-primary text-sm font-medium"
            >
                <span>{selectedModel.name}</span>
                <span className="text-xs opacity-50">▼</span>
            </button>

            <AnimatePresence>
                {isOpen && (
                    <>
                        <div
                            className="fixed inset-0 z-40"
                            onClick={() => setIsOpen(false)}
                        />
                        <motion.div
                            initial={{ opacity: 0, y: -4 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -4 }}
                            className="absolute top-full left-0 mt-2 w-64 p-1 rounded-xl border border-border-subtle bg-bg-elevated/90 backdrop-blur-xl shadow-panel z-50"
                        >
                            {MODELS.map((model) => (
                                <button
                                    key={model.id}
                                    onClick={() => {
                                        setSelectedModel(model)
                                        setIsOpen(false)
                                    }}
                                    className={`w-full text-left px-3 py-2.5 rounded-lg transition-colors flex flex-col gap-0.5 ${selectedModel.id === model.id
                                        ? 'bg-accent-primary/10'
                                        : 'hover:bg-white/5'
                                        }`}
                                >
                                    <div className="flex items-center justify-between">
                                        <span className={`text-sm font-medium ${selectedModel.id === model.id ? 'text-accent-primary' : 'text-text-primary'
                                            }`}>
                                            {model.name}
                                        </span>
                                        {selectedModel.id === model.id && (
                                            <span className="text-accent-primary text-xs">✓</span>
                                        )}
                                    </div>
                                    <span className="text-xs text-text-secondary line-clamp-1">
                                        {model.description}
                                    </span>
                                </button>
                            ))}
                        </motion.div>
                    </>
                )}
            </AnimatePresence>
        </div>
    )
}
