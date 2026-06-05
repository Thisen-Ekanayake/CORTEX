import {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  type ReactNode,
} from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import {
  getMe,
  login as apiLogin,
  register as apiRegister,
  getToken,
  setToken,
  clearToken,
  type UserProfile,
} from '../lib/backendApi'

interface AuthContextValue {
  user: UserProfile | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, displayName: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserProfile | null>(null)
  const [loading, setLoading] = useState(true)

  // On mount, restore the session from a stored token (if any).
  useEffect(() => {
    let active = true
    if (!getToken()) {
      setLoading(false)
      return
    }
    getMe()
      .then((u) => active && setUser(u))
      .catch(() => {
        clearToken()
        if (active) setUser(null)
      })
      .finally(() => active && setLoading(false))
    return () => {
      active = false
    }
  }, [])

  const login = useCallback(async (email: string, password: string) => {
    const token = await apiLogin(email, password)
    setToken(token)
    setUser(await getMe())
  }, [])

  const register = useCallback(
    async (email: string, password: string, displayName: string) => {
      await apiRegister(email, password, displayName)
      const token = await apiLogin(email, password)
      setToken(token)
      setUser(await getMe())
    },
    [],
  )

  const logout = useCallback(() => {
    clearToken()
    setUser(null)
  }, [])

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider')
  return ctx
}

/** Gate protected routes — redirects to /login while unauthenticated. */
export function RequireAuth({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth()
  const location = useLocation()

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-bg-base text-text-secondary">
        <div className="flex items-center gap-3">
          <span className="w-2 h-2 bg-accent-primary rounded-full animate-bounce [animation-delay:-0.3s]" />
          <span className="w-2 h-2 bg-accent-primary rounded-full animate-bounce [animation-delay:-0.15s]" />
          <span className="w-2 h-2 bg-accent-primary rounded-full animate-bounce" />
        </div>
      </div>
    )
  }

  if (!user) {
    return <Navigate to="/login" replace state={{ from: location }} />
  }

  return <>{children}</>
}
