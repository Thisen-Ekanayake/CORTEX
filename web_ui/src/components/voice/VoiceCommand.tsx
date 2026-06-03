import { useState, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { stt } from '../../lib/api'

interface VoiceCommandProps {
  /** Called with the transcribed text once recording stops and STT completes. */
  onTranscript: (text: string) => void
}

/**
 * Records a short audio clip via MediaRecorder and transcribes it with the
 * local Parakeet model (POST /api/stt). Replaces the old browser-native
 * SpeechRecognition approach so transcription uses CORTEX's own model.
 */
export function VoiceCommand({ onTranscript }: VoiceCommandProps) {
  const [recording, setRecording] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  const startRecording = useCallback(async () => {
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Mic not supported')
      return
    }
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      setError('Mic permission denied')
      return
    }

    const recorder = new MediaRecorder(stream)
    chunksRef.current = []
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop())
      const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
      setBusy(true)
      try {
        const text = await stt(blob)
        if (text) onTranscript(text)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Transcription failed')
      } finally {
        setBusy(false)
      }
    }
    recorderRef.current = recorder
    recorder.start()
    setRecording(true)
  }, [onTranscript])

  const stopRecording = useCallback(() => {
    recorderRef.current?.stop()
    recorderRef.current = null
    setRecording(false)
  }, [])

  return (
    <div className="voice-command">
      <button
        type="button"
        className={`voice-command__btn ${recording ? 'voice-command__btn--active' : ''}`}
        onClick={recording ? stopRecording : startRecording}
        disabled={busy}
        aria-label={recording ? 'Stop recording' : 'Start voice input'}
        aria-pressed={recording}
        title="Voice input (Parakeet)"
      >
        <span className="voice-command__mic" aria-hidden>
          {busy ? '…' : recording ? '◉' : '🎤'}
        </span>
      </button>
      <AnimatePresence>
        {recording && (
          <motion.div
            className="voice-command__feedback"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
          >
            <span className="voice-command__pulse" />
            Recording…
          </motion.div>
        )}
        {busy && (
          <motion.span
            className="voice-command__feedback"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            Transcribing…
          </motion.span>
        )}
        {error && (
          <motion.span
            className="voice-command__unsupported"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            {error}
          </motion.span>
        )}
      </AnimatePresence>
    </div>
  )
}
