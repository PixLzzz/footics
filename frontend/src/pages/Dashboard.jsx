import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getMatches, deleteMatch } from '../api'
import { Play, Trash2, Clock, Video, CheckCircle, Loader, AlertCircle, Upload, ChevronRight } from 'lucide-react'

const statusConfig = {
  uploaded: { icon: Video, color: 'text-blue-400', bg: 'bg-blue-400/10', label: 'Uploadé' },
  processing: { icon: Loader, color: 'text-yellow-400', bg: 'bg-yellow-400/10', label: 'Analyse...', spin: true },
  analyzed: { icon: CheckCircle, color: 'text-emerald-400', bg: 'bg-emerald-400/10', label: 'Analysé' },
  error: { icon: AlertCircle, color: 'text-red-400', bg: 'bg-red-400/10', label: 'Erreur' },
}

function formatDuration(s) {
  if (!s) return '--:--'
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

export default function Dashboard() {
  const [matches, setMatches] = useState([])
  const [loading, setLoading] = useState(true)

  const load = () => {
    setLoading(true)
    getMatches().then(setMatches).finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  const handleDelete = async (e, id) => {
    e.preventDefault()
    e.stopPropagation()
    if (!confirm('Supprimer ce match ?')) return
    await deleteMatch(id)
    load()
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Mes Matchs</h1>
          <p className="text-sm text-gray-500 mt-1">{matches.length} match{matches.length !== 1 && 's'} enregistré{matches.length !== 1 && 's'}</p>
        </div>
        <Link to="/upload" className="btn-primary flex items-center gap-2">
          <Upload size={16} /> Nouveau match
        </Link>
      </div>

      {loading ? (
        <div className="flex justify-center py-20">
          <Loader className="animate-spin text-blue-500" size={32} />
        </div>
      ) : matches.length === 0 ? (
        <div className="card text-center py-20">
          <div className="w-16 h-16 rounded-2xl bg-blue-600/10 flex items-center justify-center mx-auto mb-4">
            <Video size={28} className="text-blue-900" />
          </div>
          <p className="text-gray-400 text-lg mb-2">Aucun match enregistré</p>
          <p className="text-gray-600 mb-6">Commencez par uploader une vidéo de match</p>
          <Link to="/upload" className="btn-primary inline-flex items-center gap-2">
            <Upload size={16} /> Uploader un match
          </Link>
        </div>
      ) : (
        <div className="grid gap-3">
          {matches.map((m) => {
            const status = statusConfig[m.status] || statusConfig.uploaded
            const StatusIcon = status.icon
            return (
              <Link
                key={m.id}
                to={`/match/${m.id}`}
                className="card group flex items-center gap-4 hover:border-blue-700/50 transition-all cursor-pointer !py-4"
              >
                <div className={`w-10 h-10 rounded-xl ${status.bg} flex items-center justify-center shrink-0`}>
                  <StatusIcon size={18} className={`${status.color} ${status.spin ? 'animate-spin' : ''}`} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-semibold group-hover:text-blue-400 transition-colors truncate">
                    {m.title}
                  </div>
                  <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
                    <span className={`font-medium ${status.color}`}>{status.label}</span>
                    <span className="flex items-center gap-1">
                      <Clock size={11} /> {formatDuration(m.duration_seconds)}
                    </span>
                    {m.team && (
                      <span className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: m.team.color }} />
                        <span style={{ color: m.team.color }}>{m.team.name}</span>
                      </span>
                    )}
                    {m.event_count > 0 && (
                      <span>{m.event_count} événement{m.event_count !== 1 && 's'}</span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={(e) => handleDelete(e, m.id)}
                    className="p-2 rounded-lg text-gray-600 hover:text-red-400 hover:bg-red-900/20 transition-colors opacity-0 group-hover:opacity-100"
                  >
                    <Trash2 size={14} />
                  </button>
                  <ChevronRight size={16} className="text-gray-600 group-hover:text-blue-400 transition-colors" />
                </div>
              </Link>
            )
          })}
        </div>
      )}
    </div>
  )
}
