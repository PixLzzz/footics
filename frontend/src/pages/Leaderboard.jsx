import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getLeaderboard } from '../api'
import { Trophy, Loader } from 'lucide-react'

const CATEGORIES = [
  { value: 'goal', label: 'Buts' },
  { value: 'assist', label: 'Passes dé' },
  { value: 'shot', label: 'Tirs' },
  { value: 'interception', label: 'Interceptions' },
  { value: 'dribble', label: 'Dribbles' },
  { value: 'key_pass', label: 'Passes clés' },
  { value: 'pass', label: 'Passes' },
]

export default function Leaderboard() {
  const [category, setCategory] = useState('goal')
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    getLeaderboard(category).then(setData).finally(() => setLoading(false))
  }, [category])

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Trophy size={24} className="text-yellow-400" /> Classement
        </h1>
        <p className="text-sm text-gray-500 mt-1">Top joueurs par catégorie</p>
      </div>

      <div className="flex gap-2 flex-wrap mb-6">
        {CATEGORIES.map((c) => (
          <button
            key={c.value}
            onClick={() => setCategory(c.value)}
            className={`px-3.5 py-1.5 rounded-xl text-sm font-medium transition-all ${
              category === c.value
                ? 'bg-blue-600/20 text-blue-300 border border-blue-500/40 shadow-sm shadow-blue-900/20'
                : 'bg-slate-800/60 text-gray-500 hover:text-gray-300 border border-transparent'
            }`}
          >
            {c.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex justify-center py-16"><Loader className="animate-spin text-blue-500" /></div>
      ) : data.length === 0 ? (
        <div className="card text-center py-16">
          <Trophy size={36} className="mx-auto text-gray-800 mb-3" />
          <p className="text-gray-500">Aucune donnée pour cette catégorie.</p>
          <p className="text-gray-600 text-sm mt-1">Les stats apparaîtront après la détection d'événements.</p>
        </div>
      ) : (
        <div className="card p-0 overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-blue-900/30">
                <th className="text-left px-5 py-3.5 text-xs font-medium text-gray-500 uppercase tracking-wider w-12">#</th>
                <th className="text-left px-5 py-3.5 text-xs font-medium text-gray-500 uppercase tracking-wider">Joueur</th>
                <th className="text-left px-5 py-3.5 text-xs font-medium text-gray-500 uppercase tracking-wider">Equipe</th>
                <th className="text-right px-5 py-3.5 text-xs font-medium text-gray-500 uppercase tracking-wider">Total</th>
              </tr>
            </thead>
            <tbody>
              {data.map((row, i) => (
                <tr key={row.player_id} className="border-b border-blue-900/15 hover:bg-blue-950/30 transition-colors">
                  <td className="px-5 py-3.5">
                    {i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : (
                      <span className="text-gray-600 font-mono text-sm">{i + 1}</span>
                    )}
                  </td>
                  <td className="px-5 py-3.5">
                    <Link to={`/player/${row.player_id}`} className="font-medium hover:text-blue-400 transition-colors flex items-center gap-2">
                      {row.jersey_number !== null && (
                        <span className="text-[10px] font-mono bg-blue-900/40 text-blue-300/80 px-1.5 py-0.5 rounded">
                          {row.jersey_number}
                        </span>
                      )}
                      {row.name}
                    </Link>
                  </td>
                  <td className="px-5 py-3.5">
                    <span className="flex items-center gap-2 text-gray-400">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: row.team_color }} />
                      {row.team_name}
                    </span>
                  </td>
                  <td className="px-5 py-3.5 text-right">
                    <span className="font-bold text-lg text-blue-300">{row.count}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
