import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { chatStream, tts, vlm } from '../lib/api'
import { VoiceCommand } from '../components/voice/VoiceCommand'

interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  route?: string
  sources?: string[]
  imageUrl?: string
}

const INITIAL_MESSAGES: Message[] = [
  {
    id: 1,
    role: 'assistant',
    content: "Hello! I'm CORTEX, your advanced AI assistant. I can help you analyze documents, generate reports, or answer complex queries. How can I assist you today?"
  }
]

export function Chat() {
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES)
  const [input, setInput] = useState('')
  const [isThinking, setIsThinking] = useState(false)
  const [thinkingStep, setThinkingStep] = useState('')
  const [speakingId, setSpeakingId] = useState<number | null>(null)
  const [attachedImage, setAttachedImage] = useState<File | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isThinking])

  const speak = async (msg: Message) => {
    if (!msg.content.trim()) return
    audioRef.current?.pause()
    setSpeakingId(msg.id)
    try {
      const url = await tts(msg.content)
      const audio = new Audio(url)
      audioRef.current = audio
      audio.onended = () => setSpeakingId(null)
      audio.onerror = () => setSpeakingId(null)
      await audio.play()
    } catch {
      setSpeakingId(null)
    }
  }

  const patchMessage = (id: number, patch: Partial<Message>) => {
    setMessages(prev => prev.map(m => (m.id === id ? { ...m, ...patch } : m)))
  }

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault()
    if (isThinking) return
    if (!input.trim() && !attachedImage) return

    const image = attachedImage
    const prompt = input.trim() || (image ? 'Describe this image.' : '')

    const userMsg: Message = {
      id: Date.now(),
      role: 'user',
      content: prompt,
      imageUrl: image ? URL.createObjectURL(image) : undefined,
    }
    const assistantId = userMsg.id + 1
    setMessages(prev => [...prev, userMsg, { id: assistantId, role: 'assistant', content: '' }])
    setInput('')
    setAttachedImage(null)
    setIsThinking(true)

    // ── Vision path: an image is attached → Qwen2.5-VL ──
    if (image) {
      setThinkingStep('Analyzing image...')
      try {
        const description = await vlm(image, prompt)
        patchMessage(assistantId, { content: description, route: 'vlm' })
      } catch (err) {
        patchMessage(assistantId, { content: `⚠️ ${err instanceof Error ? err.message : 'Vision request failed'}` })
      } finally {
        setIsThinking(false)
      }
      return
    }

    // ── Text path: routed RAG/CHAT/META streaming ──
    setThinkingStep('Routing query...')
    let streamed = false
    await chatStream(prompt, {
      onToken: (token) => {
        if (!streamed) {
          streamed = true
          setIsThinking(false)
        }
        setMessages(prev =>
          prev.map(m => (m.id === assistantId ? { ...m, content: m.content + token } : m))
        )
      },
      onDone: (info) => {
        setIsThinking(false)
        patchMessage(assistantId, { route: info.route, sources: info.sources })
      },
      onError: (message) => {
        setIsThinking(false)
        patchMessage(assistantId, { content: `⚠️ ${message}` })
      },
    })
  }

  return (
    <div className="flex flex-col h-full relative bg-bg-base/50">
      {/* Background Decorative Element */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-accent-primary/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-[10%] right-[-5%] w-[30%] h-[30%] bg-purple-500/5 rounded-full blur-[100px]" />
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-8 scroll-smooth custom-scrollbar relative">
        <div className="max-w-4xl mx-auto space-y-10">
          {messages.filter((msg) => !(msg.role === 'assistant' && msg.content === '')).map((msg) => (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 20, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.4, ease: [0.19, 1, 0.22, 1] }}
              className={`flex gap-4 md:gap-6 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {/* Avatar for Assistant */}
              {msg.role === 'assistant' && (
                <div className="relative shrink-0">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#58a6ff] to-[#8c52ff] flex items-center justify-center shadow-lg shadow-accent-primary/20">
                    <span className="text-white font-bold text-sm">C</span>
                  </div>
                  <div className="absolute -bottom-1 -right-1 w-3.5 h-3.5 bg-[#3fb950] border-2 border-bg-base rounded-full" />
                </div>
              )}

              {/* Message Bubble */}
              <div className={`group flex flex-col gap-2 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                <div className={`
                  relative px-5 py-4 rounded-2xl md:rounded-[24px] text-sm md:text-[15px] leading-relaxed transition-all duration-300
                  ${msg.role === 'user'
                    ? 'bg-gradient-to-br from-accent-primary/90 to-blue-600/90 text-white rounded-tr-none shadow-premium-blue'
                    : 'bg-white/[0.03] backdrop-blur-md border border-white/10 text-text-primary rounded-tl-none hover:bg-white/[0.05]'
                  }
                  max-w-full lg:max-w-2xl
                `}>
                  {msg.imageUrl && (
                    <img
                      src={msg.imageUrl}
                      alt="attached"
                      className="mb-2 max-h-48 rounded-xl border border-white/10 object-cover"
                    />
                  )}
                  {msg.content}
                </div>

                {/* Route + sources (assistant, after streaming completes) */}
                {msg.role === 'assistant' && (msg.route || (msg.sources && msg.sources.length > 0)) && (
                  <div className="flex flex-col gap-1.5 px-1 max-w-full lg:max-w-2xl">
                    {msg.route && (
                      <span className="self-start text-[10px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded-md bg-accent-primary/10 text-accent-primary border border-accent-primary/20">
                        {msg.route.replace('_', ' ')}
                      </span>
                    )}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {msg.sources.map((src, i) => (
                          <span
                            key={i}
                            className="text-[11px] text-text-secondary bg-white/[0.04] border border-white/10 rounded-lg px-2 py-1"
                            title={src}
                          >
                            ◈ {src}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Message Meta + actions */}
                <div className="flex items-center gap-2 px-1">
                  <span className="text-[10px] uppercase tracking-wider text-text-tertiary font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                    {msg.role === 'assistant' ? 'CORTEX AI' : 'YOU'} • JUST NOW
                  </span>
                  {msg.role === 'assistant' && msg.content.trim() && (
                    <button
                      type="button"
                      onClick={() => speak(msg)}
                      disabled={speakingId === msg.id}
                      title="Read aloud"
                      aria-label="Read aloud"
                      className="text-[12px] text-text-tertiary hover:text-accent-primary transition-colors opacity-0 group-hover:opacity-100 disabled:opacity-100"
                    >
                      {speakingId === msg.id ? '◼ speaking…' : '🔊'}
                    </button>
                  )}
                </div>
              </div>

              {/* Avatar for User */}
              {msg.role === 'user' && (
                <div className="shrink-0">
                  <div className="w-10 h-10 rounded-xl bg-bg-elevated border border-border-subtle flex items-center justify-center shadow-md">
                    <span className="text-text-secondary font-bold text-sm">TE</span>
                  </div>
                </div>
              )}
            </motion.div>
          ))}

          {/* Thinking Indicator */}
          {isThinking && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex gap-4 md:gap-6 items-start"
            >
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#58a6ff]/50 to-[#8c52ff]/50 flex items-center justify-center animate-pulse shrink-0">
                <span className="text-white/70 font-bold text-sm">C</span>
              </div>
              <div className="flex flex-col gap-3 pt-2">
                <div className="flex items-center gap-3">
                  <div className="flex gap-1">
                    <span className="w-1.5 h-1.5 bg-accent-primary rounded-full animate-bounce [animation-delay:-0.3s]" />
                    <span className="w-1.5 h-1.5 bg-accent-primary rounded-full animate-bounce [animation-delay:-0.15s]" />
                    <span className="w-1.5 h-1.5 bg-accent-primary rounded-full animate-bounce" />
                  </div>
                  <span className="text-sm font-medium text-accent-primary animate-pulse">{thinkingStep}</span>
                </div>
                <div className="w-48 h-1 bg-white/5 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-accent-primary"
                    initial={{ width: "0%" }}
                    animate={{ width: "100%" }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                </div>
              </div>
            </motion.div>
          )}
          <div ref={bottomRef} className="h-4" />
        </div>
      </div>

      {/* Input Area */}
      <div className="px-4 pb-8 pt-2 md:px-8 md:pb-10 relative z-10">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel rounded-3xl p-2 md:p-3 shadow-2xl relative overflow-hidden group"
          >
            {/* Inner Glow focus effect */}
            <div className="absolute inset-0 bg-accent-primary/0 group-focus-within:bg-accent-primary/[0.02] transition-colors pointer-events-none" />

            <form onSubmit={handleSend} className="relative flex flex-col gap-2">
              {attachedImage && (
                <div className="flex items-center gap-2 px-3 pt-1">
                  <span className="text-[11px] text-accent-primary bg-accent-primary/10 border border-accent-primary/20 rounded-lg px-2 py-1">
                    🖼 {attachedImage.name}
                  </span>
                  <button
                    type="button"
                    onClick={() => setAttachedImage(null)}
                    className="text-[11px] text-text-tertiary hover:text-text-primary"
                    aria-label="Remove attached image"
                  >
                    ✕ remove
                  </button>
                </div>
              )}
              <div className="flex items-end gap-2 px-2">
                <input
                  ref={imageInputRef}
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const f = e.target.files?.[0]
                    if (f) setAttachedImage(f)
                    if (imageInputRef.current) imageInputRef.current.value = ''
                  }}
                />
                <button
                  type="button"
                  onClick={() => imageInputRef.current?.click()}
                  className="p-2.5 text-text-tertiary hover:text-text-primary transition-all rounded-xl hover:bg-white/5 shrink-0"
                  aria-label="Attach image"
                  title="Attach image (Vision)"
                >
                  <span className="text-xl">＋</span>
                </button>

                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSend(e);
                    }
                  }}
                  placeholder="Message CORTEX..."
                  rows={1}
                  className="flex-1 bg-transparent text-text-primary placeholder:text-text-tertiary py-3 px-1 resize-none outline-none max-h-60 min-h-[44px] text-base leading-relaxed custom-scrollbar"
                />

                <button
                  type="submit"
                  disabled={!input.trim() || isThinking}
                  className={`
                    mb-1 p-2.5 rounded-xl transition-all duration-300 transform active:scale-95
                    ${input.trim() && !isThinking
                      ? 'bg-accent-primary text-white shadow-glow hover:translate-y-[-2px]'
                      : 'bg-white/5 text-text-tertiary'
                    }
                  `}
                >
                  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13"></line>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                  </svg>
                </button>
              </div>

              {/* Toolbar Bottom */}
              <div className="flex items-center justify-between px-3 pb-1 border-t border-white/[0.03] pt-2">
                <div className="flex gap-3 items-center">
                  <VoiceCommand
                    onTranscript={(text) =>
                      setInput((prev) => (prev ? `${prev} ${text}` : text))
                    }
                  />
                  <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-lg hover:bg-white/5 text-[11px] text-text-tertiary transition-colors cursor-pointer">
                    <span className="w-1.5 h-1.5 border border-text-tertiary rounded-sm" />
                    <span>SEARCH</span>
                  </div>
                  <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-lg hover:bg-white/5 text-[11px] text-text-tertiary transition-colors cursor-pointer">
                    <span className="w-1.5 h-1.5 border border-text-tertiary rounded-sm" />
                    <span>ANALYZE</span>
                  </div>
                </div>
                <div className="text-[10px] text-text-tertiary font-medium">
                  ENTER TO SEND • SHIFT+ENTER FOR NEW LINE
                </div>
              </div>
            </form>
          </motion.div>

          <div className="text-center mt-4">
            <p className="text-[11px] tracking-wide text-text-tertiary uppercase font-medium opacity-60">
              CORTEX Professional • Secure Enterprise Environment
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
