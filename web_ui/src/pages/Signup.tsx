import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useAuth } from '../contexts/AuthContext'

export function Signup() {
  const { register } = useAuth()
  const navigate = useNavigate()

  const [displayName, setDisplayName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (submitting) return
    setError(null)
    if (password.length < 8) {
      setError('Password must be at least 8 characters.')
      return
    }
    setSubmitting(true)
    try {
      await register(email.trim(), password, displayName.trim())
      navigate('/', { replace: true })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign up failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="flex h-screen items-center justify-center bg-bg-base text-text-primary relative overflow-hidden">
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-accent-primary/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-[5%] right-[-5%] w-[35%] h-[35%] bg-purple-500/10 rounded-full blur-[120px]" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="glass-panel rounded-3xl p-8 w-full max-w-md relative z-10 shadow-2xl"
      >
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-lg bg-accent-primary/20 flex items-center justify-center text-accent-primary font-bold">
            C
          </div>
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Create your account</h1>
            <p className="text-sm text-text-tertiary">Get started with CORTEX</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1.5 uppercase tracking-wider">
              Display name
            </label>
            <input
              type="text"
              required
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              autoComplete="name"
              className="w-full bg-white/[0.03] border border-white/10 rounded-xl px-4 py-3 text-sm outline-none focus:border-accent-primary/50 focus:bg-white/[0.05] transition-colors"
              placeholder="Alex Morgan"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1.5 uppercase tracking-wider">
              Email
            </label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
              className="w-full bg-white/[0.03] border border-white/10 rounded-xl px-4 py-3 text-sm outline-none focus:border-accent-primary/50 focus:bg-white/[0.05] transition-colors"
              placeholder="you@example.com"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1.5 uppercase tracking-wider">
              Password
            </label>
            <input
              type="password"
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="new-password"
              className="w-full bg-white/[0.03] border border-white/10 rounded-xl px-4 py-3 text-sm outline-none focus:border-accent-primary/50 focus:bg-white/[0.05] transition-colors"
              placeholder="At least 8 characters"
            />
          </div>

          {error && (
            <p className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="w-full bg-accent-primary text-white font-medium rounded-xl py-3 shadow-glow hover:translate-y-[-1px] transition-all disabled:opacity-60 disabled:translate-y-0"
          >
            {submitting ? 'Creating account…' : 'Create account'}
          </button>
        </form>

        <p className="text-sm text-text-tertiary text-center mt-6">
          Already have an account?{' '}
          <Link to="/login" className="text-accent-primary hover:underline">
            Sign in
          </Link>
        </p>
      </motion.div>
    </div>
  )
}
