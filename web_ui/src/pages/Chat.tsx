import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'

// Placeholder data
const INITIAL_MESSAGES = [
  {
    id: 1,
    role: 'assistant',
    content: "Hello! I'm CORTEX, your advanced AI assistant. I can help you analyze documents, generate reports, or answer complex queries. How can I assist you today?"
  }
]

export function Chat() {
  const [messages, setMessages] = useState(INITIAL_MESSAGES)
  const [input, setInput] = useState('')
  const [isThinking, setIsThinking] = useState(false)
  const [thinkingStep, setThinkingStep] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isThinking])

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isThinking) return

    const userMsg = { id: Date.now(), role: 'user', content: input.trim() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setIsThinking(true)
    setThinkingStep('Analyzing request...')

    // Simulate AI processing
    setTimeout(() => setThinkingStep('Searching knowledge base...'), 800)
    setTimeout(() => setThinkingStep('Generating response...'), 1600)

    setTimeout(() => {
      setIsThinking(false)
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        content: "I've analyzed your request. Based on the available data, I can confirm that the Q3 financial report shows a 15% increase in revenue. Would you like a detailed breakdown?"
      }])
    }, 2500)
  }

  return (
    <div className="flex flex-col h-full relative">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 scroll-smooth">
        {messages.map((msg) => (
          <motion.div
            key={msg.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start max-w-3xl mx-auto'}`}
          >
            {/* Avatar for Assistant */}
            {msg.role === 'assistant' && (
              <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-accent-primary to-purple-600 flex items-center justify-center text-xs text-white font-bold shadow-glow shrink-0">
                C
              </div>
            )}

            {/* Message Bubble */}
            <div className={`max-w-[85%] lg:max-w-[75%] space-y-1 ${msg.role === 'user' ? 'order-1' : 'order-2'}`}>
              <div className={`
                p-3.5 md:p-4 rounded-2xl text-sm md:text-base leading-relaxed shadow-sm
                ${msg.role === 'user'
                  ? 'bg-bg-elevated text-text-primary rounded-tr-sm border border-border-subtle'
                  : 'bg-transparent text-text-primary px-0 py-0'
                }
              `}>
                {msg.content}
              </div>
            </div>

            {/* Avatar for User (Hidden but keeps layout balanced if needed, or just omit) */}
            {msg.role === 'user' && (
              <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center text-xs text-white font-bold shrink-0 order-2">
                TE
              </div>
            )}
          </motion.div>
        ))}

        {/* Thinking Indicator */}
        {isThinking && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex gap-4 max-w-3xl mx-auto"
          >
            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-accent-primary to-purple-600 flex items-center justify-center text-xs text-white font-bold shadow-glow shrink-0 animate-pulse">
              C
            </div>
            <div className="flex items-center gap-3 text-sm text-text-secondary">
              <span className="w-2 h-2 bg-accent-primary rounded-full animate-bounce" />
              <span>{thinkingStep}</span>
            </div>
          </motion.div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 md:p-6 bg-bg-base/80 backdrop-blur-md sticky bottom-0 z-10">
        <div className="max-w-3xl mx-auto relative group">
          <form
            onSubmit={handleSend}
            className="relative bg-bg-elevated/50 border border-border-subtle focus-within:border-accent-primary/50 focus-within:ring-1 focus-within:ring-accent-primary/50 rounded-xl transition-all shadow-lg"
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend(e);
                }
              }}
              placeholder="Ask anything..."
              rows={1}
              className="w-full bg-transparent text-text-primary placeholder:text-text-tertiary p-3 md:p-4 pr-12 md:pr-14 resize-none outline-none max-h-48 min-h-[52px] custom-scrollbar"
              style={{ minHeight: '52px' }} // default height
            />

            {/* Attachments / Actions (Left) - Future implementation */}
            <button
              type="button"
              className="absolute left-3 bottom-3 p-1.5 text-text-tertiary hover:text-text-primary transition-colors rounded-md hover:bg-white/5"
              aria-label="Attach file"
            >
              <span className="text-lg">＋</span>
            </button>

            {/* Input Padding Adjustment for Left Icon */}
            <style>{`
              textarea { padding-left: 40px !important; }
            `}</style>

            {/* Send Button (Right) */}
            <button
              type="submit"
              disabled={!input.trim() || isThinking}
              className="absolute right-2 bottom-2 p-2 rounded-lg bg-accent-primary text-white disabled:opacity-50 disabled:bg-transparent disabled:text-text-tertiary transition-all hover:bg-accent-primary/90"
            >
              <span className="text-sm font-semibold">↑</span>
            </button>
          </form>
          <div className="text-center mt-2">
            <p className="text-xs text-text-tertiary">
              CORTEX can make mistakes. Verify important information.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
