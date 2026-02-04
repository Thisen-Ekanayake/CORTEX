import { useState, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export function VoiceCommand() {
  const [listening, setListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [supported, setSupported] = useState<boolean | null>(null)
  const recognitionRef = useRef<SpeechRecognition | null>(null)

  const startListening = useCallback(() => {
    const Win = window as unknown as { webkitSpeechRecognition?: new () => SpeechRecognition; SpeechRecognition?: new () => SpeechRecognition }
    if (!Win.webkitSpeechRecognition && !Win.SpeechRecognition) {
      setSupported(false)
      return
    }
    setSupported(true)
    const SR = Win.webkitSpeechRecognition ?? Win.SpeechRecognition
    if (!SR) {
      setSupported(false)
      return
    }
    const recognition = new SR()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'
    recognition.onresult = (e: SpeechRecognitionEvent) => {
      const last = e.results.length - 1
      const text = (e.results[last] as SpeechRecognitionResult)[0].transcript
      setTranscript(text)
    }
    recognition.onend = () => {
      setListening(false)
      recognitionRef.current = null
    }
    recognitionRef.current = recognition
    recognition.start()
    setListening(true)
    setTranscript('')
  }, [])

  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
    }
    setListening(false)
  }, [])

  return (
    <div className="voice-command">
      <button
        type="button"
        className={`voice-command__btn ${listening ? 'voice-command__btn--active' : ''}`}
        onClick={listening ? stopListening : startListening}
        aria-label={listening ? 'Stop voice command' : 'Start voice command'}
        aria-pressed={listening}
        title="Voice command (optional)"
      >
        <span className="voice-command__mic" aria-hidden>
          {listening ? '◉' : '⌤'}
        </span>
      </button>
      <AnimatePresence>
        {listening && (
          <motion.div
            className="voice-command__feedback"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
          >
            <span className="voice-command__pulse" />
            Listening…
            {transcript && <span className="voice-command__transcript">{transcript}</span>}
          </motion.div>
        )}
        {supported === false && (
          <motion.span
            className="voice-command__unsupported"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            Voice not supported
          </motion.span>
        )}
      </AnimatePresence>
    </div>
  )
}
