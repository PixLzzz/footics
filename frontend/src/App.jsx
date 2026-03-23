import { Routes, Route, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import TeamsPlayers from './pages/TeamsPlayers'
import MatchUpload from './pages/MatchUpload'
import MatchDetail from './pages/MatchDetail'
import PlayerStats from './pages/PlayerStats'
import Leaderboard from './pages/Leaderboard'
import { LayoutDashboard, Upload, Users, Trophy } from 'lucide-react'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Matchs' },
  { to: '/upload', icon: Upload, label: 'Upload' },
  { to: '/teams', icon: Users, label: 'Equipes' },
  { to: '/leaderboard', icon: Trophy, label: 'Classement' },
]

export default function App() {
  return (
    <div className="min-h-screen flex bg-slate-950">
      {/* Sidebar */}
      <nav className="w-56 bg-gradient-to-b from-slate-900 to-slate-950 border-r border-blue-900/20 flex flex-col p-4 gap-1 shrink-0">
        <div className="flex items-center gap-2.5 px-3 py-4 mb-3">
          <span className="text-2xl">⚽</span>
          <div className="flex flex-col leading-tight">
            <span className="text-lg font-extrabold tracking-tight">
              <span className="text-blue-400">Foot</span>
              <span className="text-red-500">ics</span>
            </span>
            <span className="text-[9px] text-gray-600 tracking-widest uppercase">Analytics</span>
          </div>
        </div>
        <div className="w-full h-px bg-gradient-to-r from-blue-600/40 via-red-500/30 to-transparent mb-3" />
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
                isActive
                  ? 'bg-blue-600/15 text-blue-400 shadow-sm shadow-blue-900/20'
                  : 'text-gray-500 hover:text-gray-200 hover:bg-slate-800/60'
              }`
            }
          >
            <Icon size={17} />
            {label}
          </NavLink>
        ))}
        <div className="mt-auto pt-4">
          <div className="w-full h-px bg-gradient-to-r from-blue-600/30 via-red-500/20 to-transparent mb-3" />
          <p className="text-[10px] text-gray-700 text-center">Footics Analytics</p>
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto p-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<MatchUpload />} />
            <Route path="/teams" element={<TeamsPlayers />} />
            <Route path="/match/:id" element={<MatchDetail />} />
            <Route path="/player/:id" element={<PlayerStats />} />
            <Route path="/leaderboard" element={<Leaderboard />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}
