import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Button } from '../components/ui/Button'
import { chatMessages } from '../data/placeholder'
import './pages.css'

function TypedMessage({ text, delay = 0 }: { text: string; delay?: number }) {
  const [displayed, setDisplayed] = useState('')
  const [done, setDone] = useState(false)

  useEffect(() => {
    const start = Date.now() + delay
    let i = 0
    const t = setInterval(() => {
      if (Date.now() < start) return
      if (i >= text.length) {
        setDone(true)
        clearInterval(t)
        return
      }
      setDisplayed(text.slice(0, i + 1))
      i++
    }, 16)
    return () => clearInterval(t)
  }, [text, delay])

  return <span>{done ? text : displayed}</span>
}

export function Chat() {
  const [messages, setMessages] = useState(chatMessages)
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = () => {
    if (!input.trim() || sending) return
    const userMsg = { role: 'user' as const, content: input.trim() }
    setMessages((m) => [...m, userMsg])
    setInput('')
    setSending(true)
    setTimeout(() => {
      setMessages((m) => [
        ...m,
        {
          role: 'assistant' as const,
          content: "I'm the CORTEX assistant. In a full implementation, your query would be routed (RAG / CHAT / META) and answered from your knowledge base. Try the Search or Documents pages for more.",
        },
      ])
      setSending(false)
    }, 800)
  }

  return (
    <motion.div
      className="page page--chat"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <h2 className="page__title">Assist</h2>
      <p className="page__subtitle">
        Integrated assistant with smooth message transitions and typed animations.
      </p>

      <div className="chat-panel">
        <div className="chat-messages">
          <AnimatePresence initial={false}>
            {messages.map((msg, i) => (
              <motion.div
                key={i}
                className={`chat-message chat-message--${msg.role}`}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25 }}
              >
                <div className="chat-message__bubble">
                  {msg.role === 'assistant' && i === 1 ? (
                    <TypedMessage text={msg.content} delay={200} />
                  ) : (
                    msg.content
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          {sending && (
            <motion.div
              className="chat-message chat-message--assistant"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="chat-message__bubble chat-message__bubble--typing">
                <span className="typing-dot" />
                <span className="typing-dot" />
                <span className="typing-dot" />
              </div>
            </motion.div>
          )}
          <div ref={bottomRef} />
        </div>

        <form
          className="chat-input-wrap"
          onSubmit={(e) => {
            e.preventDefault()
            send()
          }}
        >
          <input
            type="text"
            className="chat-input"
            placeholder="Ask anything…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={sending}
            aria-label="Message"
          />
          <Button type="submit" disabled={sending || !input.trim()} aria-label="Send">
            Send
          </Button>
        </form>
      </div>
    </motion.div>
  )
}
