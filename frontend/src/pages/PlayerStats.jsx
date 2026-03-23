import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getPlayerStats } from '../api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { ArrowLeft, Loader } from 'lucide-react'

const STAT_LABELS = {
  goal: 'Buts',
  assist: 'Passes dé',
  shot: 'Tirs',
  shot_on_target: 'Tirs cadrés',
  pass: 'Passes',
  key_pass: 'Passes clés',
  interception: 'Interceptions',
  tackle: 'Tacles',
  foul: 'Fautes',
  save: 'Arrêts',
  dribble: 'Dribbles',
}

const STAT_COLORS = {
  goal: '#3b82f6',
  assist: '#ef4444',
  shot: '#2563eb',
  shot_on_target: '#f87171',
  pass: '#60a5fa',
  key_pass: '#818cf8',
  interception: '#1e40af',
  tackle: '#b91c1c',
  foul: '#f59e0b',
  save: '#34d399',
  dribble: '#fb923c',
}

export default function PlayerStats() {
  const { id } = useParams()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    getPlayerStats(id).then(setData).finally(() => setLoading(false))
  }, [id])

  if (loading) return <div className="flex justify-center py-20"><Loader className="animate-spin text-blue-500" /></div>
  if (!data) return <p>Joueur introuvable</p>

  const chartData = Object.entries(data.stats).map(([type, count]) => ({
    name: STAT_LABELS[type] || type,
    value: count,
    color: STAT_COLORS[type] || '#6b7280',
  })).sort((a, b) => b.value - a.value)

  const { player } = data

  return (
    <div>
      <Link to="/teams" className="text-gray-500 hover:text-blue-400 text-sm flex items-center gap-1 mb-6 transition-colors">
        <ArrowLeft size={14} /> Retour aux équipes
      </Link>

      <div className="flex items-center gap-5 mb-8">
        <div
          className="w-16 h-16 rounded-2xl flex items-center justify-center text-xl font-bold"
          style={{ backgroundColor: player.team_color + '20', color: player.team_color }}
        >
          {player.jersey_number ?? '?'}
        </div>
        <div>
          <h1 className="text-2xl font-bold">{player.name}</h1>
          <p className="text-gray-500 text-sm mt-0.5">
            <span style={{ color: player.team_color }}>{player.team_name}</span>
            {' '}&middot; {data.matches_played} match{data.matches_played !== 1 && 's'} joué{data.matches_played !== 1 && 's'}
          </p>
        </div>
      </div>

      {chartData.length === 0 ? (
        <div className="card text-center py-16">
          <p className="text-gray-500">Aucune statistique enregistrée pour ce joueur.</p>
          <p className="text-gray-600 text-sm mt-1">Les stats apparaîtront quand des événements seront ajoutés.</p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 mb-6">
            {chartData.map((s) => (
              <div key={s.name} className="card text-center !py-5">
                <p className="text-3xl font-bold" style={{ color: s.color }}>{s.value}</p>
                <p className="text-xs text-gray-500 mt-1.5">{s.name}</p>
              </div>
            ))}
          </div>

          <div className="card">
            <h3 className="font-semibold mb-4 text-sm text-gray-400">Répartition</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 80 }}>
                <XAxis type="number" stroke="#334155" tick={{ fill: '#64748b', fontSize: 12 }} />
                <YAxis type="category" dataKey="name" stroke="#334155" width={80} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    border: '1px solid rgba(30,58,95,0.5)',
                    borderRadius: 12,
                    boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
                  }}
                  labelStyle={{ color: '#e2e8f0' }}
                />
                <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                  {chartData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  )
}
