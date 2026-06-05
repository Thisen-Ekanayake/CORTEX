import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from './components/layout/Layout'
import { Chat } from './pages/Chat'
import { Home } from './pages/Home'
import { Search } from './pages/Search'
import { Documents } from './pages/Documents'
import { Settings } from './pages/Settings'
import { Project } from './pages/Project'
import { Login } from './pages/Login'
import { Signup } from './pages/Signup'
import { AuthProvider, RequireAuth } from './contexts/AuthContext'
import { ConversationsProvider } from './contexts/ConversationsContext'

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          {/* Public auth routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />

          {/* Protected app */}
          <Route
            element={
              <RequireAuth>
                <ConversationsProvider>
                  <Layout />
                </ConversationsProvider>
              </RequireAuth>
            }
          >
            <Route path="/" element={<Home />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/chat/new" element={<Chat />} />
            <Route path="/chat/:id" element={<Chat />} />
            <Route path="/project/:id" element={<Project />} />
            <Route path="/search" element={<Search />} />
            <Route path="/documents" element={<Documents />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}
