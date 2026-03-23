import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  getTeams, createTeam, updateTeam, deleteTeam,
  getPlayers, createPlayer, updatePlayer, deletePlayer,
} from '../api'
import {
  Plus, Trash2, Users, UserPlus, BarChart3, Pencil, X, Check, Shield,
} from 'lucide-react'

const POSITIONS = ['Gardien', 'Défenseur', 'Milieu', 'Attaquant', 'Pivot']

// ── Inline edit modal ────────────────────────────────────────────────────────

function Modal({ open, onClose, title, children }) {
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div
        className="bg-slate-900 border border-blue-900/50 rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden animate-in"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-blue-900/30">
          <h3 className="font-semibold text-lg">{title}</h3>
          <button onClick={onClose} className="text-gray-500 hover:text-white transition-colors">
            <X size={18} />
          </button>
        </div>
        <div className="px-6 py-5">{children}</div>
      </div>
    </div>
  )
}

export default function TeamsPlayers() {
  const [teams, setTeams] = useState([])
  const [players, setPlayers] = useState([])
  const [filterTeam, setFilterTeam] = useState('')

  // Create forms
  const [newTeam, setNewTeam] = useState('')
  const [teamColor, setTeamColor] = useState('#1D4ED8')
  const [selectedTeam, setSelectedTeam] = useState('')
  const [newPlayer, setNewPlayer] = useState('')
  const [jersey, setJersey] = useState('')
  const [position, setPosition] = useState('')

  // Errors
  const [teamError, setTeamError] = useState('')
  const [playerError, setPlayerError] = useState('')

  // Edit modals
  const [editingTeam, setEditingTeam] = useState(null)
  const [editTeamName, setEditTeamName] = useState('')
  const [editTeamColor, setEditTeamColor] = useState('')
  const [editTeamError, setEditTeamError] = useState('')

  const [editingPlayer, setEditingPlayer] = useState(null)
  const [editPlayerName, setEditPlayerName] = useState('')
  const [editPlayerJersey, setEditPlayerJersey] = useState('')
  const [editPlayerTeam, setEditPlayerTeam] = useState('')
  const [editPlayerPosition, setEditPlayerPosition] = useState('')
  const [editPlayerError, setEditPlayerError] = useState('')

  const load = async () => {
    const t = await getTeams()
    setTeams(t)
    const p = await getPlayers()
    setPlayers(p)
  }

  useEffect(() => { load() }, [])

  const filteredPlayers = filterTeam
    ? players.filter((p) => String(p.team_id) === filterTeam)
    : players

  // ── Team CRUD ──────────────────────────────────────────────────────────

  const handleAddTeam = async (e) => {
    e.preventDefault()
    if (!newTeam.trim()) return
    setTeamError('')
    try {
      await createTeam({ name: newTeam, color: teamColor })
      setNewTeam('')
      load()
    } catch (err) {
      setTeamError(err.message)
    }
  }

  const openEditTeam = (team) => {
    setEditingTeam(team)
    setEditTeamName(team.name)
    setEditTeamColor(team.color)
    setEditTeamError('')
  }

  const handleUpdateTeam = async (e) => {
    e.preventDefault()
    if (!editTeamName.trim()) return
    setEditTeamError('')
    try {
      await updateTeam(editingTeam.id, { name: editTeamName, color: editTeamColor })
      setEditingTeam(null)
      load()
    } catch (err) {
      setEditTeamError(err.message)
    }
  }

  const handleDeleteTeam = async (id) => {
    if (!confirm('Supprimer cette équipe et tous ses joueurs ?')) return
    await deleteTeam(id)
    load()
  }

  // ── Player CRUD ────────────────────────────────────────────────────────

  const handleAddPlayer = async (e) => {
    e.preventDefault()
    if (!newPlayer.trim() || !selectedTeam) return
    setPlayerError('')
    try {
      await createPlayer({
        name: newPlayer,
        team_id: selectedTeam,
        jersey_number: jersey || null,
        position,
      })
      setNewPlayer('')
      setJersey('')
      setPosition('')
      load()
    } catch (err) {
      setPlayerError(err.message)
    }
  }

  const openEditPlayer = (player) => {
    setEditingPlayer(player)
    setEditPlayerName(player.name)
    setEditPlayerJersey(player.jersey_number ?? '')
    setEditPlayerTeam(String(player.team_id))
    setEditPlayerPosition(player.position || '')
    setEditPlayerError('')
  }

  const handleUpdatePlayer = async (e) => {
    e.preventDefault()
    if (!editPlayerName.trim() || !editPlayerTeam) return
    setEditPlayerError('')
    try {
      await updatePlayer(editingPlayer.id, {
        name: editPlayerName,
        team_id: editPlayerTeam,
        jersey_number: editPlayerJersey || null,
        position: editPlayerPosition,
      })
      setEditingPlayer(null)
      load()
    } catch (err) {
      setEditPlayerError(err.message)
    }
  }

  const handleDeletePlayer = async (id) => {
    if (!confirm('Supprimer ce joueur ?')) return
    await deletePlayer(id)
    load()
  }

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Equipes & Joueurs</h1>
          <p className="text-sm text-gray-500 mt-1">
            {teams.length} équipe{teams.length !== 1 && 's'} &middot; {players.length} joueur{players.length !== 1 && 's'}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* ── Teams ──────────────────────────────────────────────────── */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-5 flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-blue-600/20 flex items-center justify-center">
              <Shield size={16} className="text-blue-400" />
            </div>
            Equipes
          </h2>

          <form onSubmit={handleAddTeam} className="mb-5">
            <div className="flex gap-2">
              <input
                type="text"
                className="input flex-1"
                placeholder="Nouvelle équipe..."
                value={newTeam}
                onChange={(e) => setNewTeam(e.target.value)}
              />
              <input
                type="color"
                className="w-10 h-10 rounded-lg cursor-pointer bg-transparent border border-blue-900/50"
                value={teamColor}
                onChange={(e) => setTeamColor(e.target.value)}
              />
              <button type="submit" className="btn-primary btn-sm flex items-center gap-1.5" disabled={!newTeam.trim()}>
                <Plus size={15} /> Ajouter
              </button>
            </div>
            {teamError && <p className="text-red-400 text-xs mt-2">{teamError}</p>}
          </form>

          <div className="space-y-2">
            {teams.map((t) => (
              <div
                key={t.id}
                className="group flex items-center justify-between px-4 py-3 bg-slate-800/60 hover:bg-slate-800 rounded-xl transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div
                    className="w-4 h-4 rounded-full ring-2 ring-offset-1 ring-offset-slate-900"
                    style={{ backgroundColor: t.color, ringColor: t.color }}
                  />
                  <div>
                    <span className="font-medium">{t.name}</span>
                    <span className="text-xs text-gray-500 ml-2">
                      {t.player_count} joueur{t.player_count !== 1 && 's'}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => openEditTeam(t)}
                    className="p-1.5 rounded-lg text-gray-500 hover:text-blue-400 hover:bg-blue-900/30 transition-colors"
                  >
                    <Pencil size={13} />
                  </button>
                  <button
                    onClick={() => handleDeleteTeam(t.id)}
                    className="p-1.5 rounded-lg text-gray-500 hover:text-red-400 hover:bg-red-900/20 transition-colors"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>
            ))}
            {teams.length === 0 && (
              <div className="text-center py-8">
                <Shield size={32} className="mx-auto text-blue-900/60 mb-2" />
                <p className="text-gray-500 text-sm">Aucune équipe</p>
                <p className="text-gray-600 text-xs mt-1">Créez votre première équipe ci-dessus</p>
              </div>
            )}
          </div>
        </div>

        {/* ── Players ────────────────────────────────────────────────── */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-5 flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-red-600/20 flex items-center justify-center">
              <Users size={16} className="text-red-400" />
            </div>
            Joueurs
          </h2>

          <form onSubmit={handleAddPlayer} className="space-y-2 mb-5" noValidate>
            <div className="flex gap-2">
              <input
                type="text"
                className="input flex-1"
                placeholder="Nom du joueur"
                value={newPlayer}
                onChange={(e) => setNewPlayer(e.target.value)}
              />
              <input
                type="number"
                className="input w-20"
                placeholder="N°"
                value={jersey}
                onChange={(e) => setJersey(e.target.value)}
              />
            </div>
            <div className="flex gap-2">
              <select
                className="input flex-1"
                value={selectedTeam}
                onChange={(e) => setSelectedTeam(e.target.value)}
              >
                <option value="">-- Equipe --</option>
                {teams.map((t) => (
                  <option key={t.id} value={t.id}>{t.name}</option>
                ))}
              </select>
              <select
                className="input flex-1"
                value={position}
                onChange={(e) => setPosition(e.target.value)}
              >
                <option value="">-- Poste --</option>
                {POSITIONS.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
              <button type="submit" className="btn-primary btn-sm flex items-center gap-1.5" disabled={!newPlayer || !selectedTeam}>
                <Plus size={15} /> Ajouter
              </button>
            </div>
            {playerError && <p className="text-red-400 text-xs">{playerError}</p>}
          </form>

          {/* Filter by team */}
          {teams.length > 0 && (
            <div className="flex gap-1.5 flex-wrap mb-4">
              <button
                onClick={() => setFilterTeam('')}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
                  !filterTeam ? 'bg-blue-600/30 text-blue-300 border border-blue-500/40' : 'bg-slate-800 text-gray-500 hover:text-gray-300'
                }`}
              >
                Tous
              </button>
              {teams.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setFilterTeam(String(t.id))}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 ${
                    filterTeam === String(t.id)
                      ? 'bg-blue-600/30 text-blue-300 border border-blue-500/40'
                      : 'bg-slate-800 text-gray-500 hover:text-gray-300'
                  }`}
                >
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: t.color }} />
                  {t.name}
                </button>
              ))}
            </div>
          )}

          <div className="space-y-1.5 max-h-[500px] overflow-y-auto pr-1">
            {filteredPlayers.map((p) => (
              <div
                key={p.id}
                className="group flex items-center justify-between px-4 py-3 bg-slate-800/60 hover:bg-slate-800 rounded-xl transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div
                    className="w-9 h-9 rounded-full flex items-center justify-center text-xs font-bold shrink-0"
                    style={{ backgroundColor: p.team_color + '25', color: p.team_color }}
                  >
                    {p.jersey_number ?? '?'}
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <Link to={`/player/${p.id}`} className="font-medium hover:text-blue-400 transition-colors truncate">
                        {p.name}
                      </Link>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                      <span>{p.team_name}</span>
                      {p.position && (
                        <>
                          <span className="text-gray-700">&middot;</span>
                          <span>{p.position}</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Link
                    to={`/player/${p.id}`}
                    className="p-1.5 rounded-lg text-gray-500 hover:text-blue-400 hover:bg-blue-900/30 transition-colors"
                  >
                    <BarChart3 size={13} />
                  </Link>
                  <button
                    onClick={() => openEditPlayer(p)}
                    className="p-1.5 rounded-lg text-gray-500 hover:text-blue-400 hover:bg-blue-900/30 transition-colors"
                  >
                    <Pencil size={13} />
                  </button>
                  <button
                    onClick={() => handleDeletePlayer(p.id)}
                    className="p-1.5 rounded-lg text-gray-500 hover:text-red-400 hover:bg-red-900/20 transition-colors"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>
            ))}
            {filteredPlayers.length === 0 && (
              <div className="text-center py-8">
                <Users size={32} className="mx-auto text-red-900/60 mb-2" />
                <p className="text-gray-500 text-sm">Aucun joueur</p>
                <p className="text-gray-600 text-xs mt-1">
                  {players.length > 0 ? 'Aucun joueur dans ce filtre' : 'Ajoutez des joueurs ci-dessus'}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Edit Team Modal ──────────────────────────────────────────── */}
      <Modal
        open={!!editingTeam}
        onClose={() => setEditingTeam(null)}
        title="Modifier l'équipe"
      >
        <form onSubmit={handleUpdateTeam} className="space-y-4">
          <div>
            <label className="label">Nom</label>
            <input
              type="text"
              className="input"
              value={editTeamName}
              onChange={(e) => setEditTeamName(e.target.value)}
              autoFocus
            />
          </div>
          <div>
            <label className="label">Couleur</label>
            <div className="flex items-center gap-3">
              <input
                type="color"
                className="w-12 h-10 rounded-lg cursor-pointer bg-transparent border border-blue-900/50"
                value={editTeamColor}
                onChange={(e) => setEditTeamColor(e.target.value)}
              />
              <span className="text-sm text-gray-400 font-mono">{editTeamColor}</span>
            </div>
          </div>
          {editTeamError && <p className="text-red-400 text-xs">{editTeamError}</p>}
          <div className="flex gap-2 justify-end pt-2">
            <button type="button" onClick={() => setEditingTeam(null)} className="btn-secondary btn-sm">
              Annuler
            </button>
            <button type="submit" className="btn-primary btn-sm flex items-center gap-1.5">
              <Check size={14} /> Enregistrer
            </button>
          </div>
        </form>
      </Modal>

      {/* ── Edit Player Modal ────────────────────────────────────────── */}
      <Modal
        open={!!editingPlayer}
        onClose={() => setEditingPlayer(null)}
        title="Modifier le joueur"
      >
        <form onSubmit={handleUpdatePlayer} className="space-y-4">
          <div className="grid grid-cols-3 gap-3">
            <div className="col-span-2">
              <label className="label">Nom</label>
              <input
                type="text"
                className="input"
                value={editPlayerName}
                onChange={(e) => setEditPlayerName(e.target.value)}
                autoFocus
              />
            </div>
            <div>
              <label className="label">N° maillot</label>
              <input
                type="number"
                className="input"
                value={editPlayerJersey}
                onChange={(e) => setEditPlayerJersey(e.target.value)}
                placeholder="--"
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="label">Equipe</label>
              <select
                className="input"
                value={editPlayerTeam}
                onChange={(e) => setEditPlayerTeam(e.target.value)}
              >
                {teams.map((t) => (
                  <option key={t.id} value={t.id}>{t.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="label">Poste</label>
              <select
                className="input"
                value={editPlayerPosition}
                onChange={(e) => setEditPlayerPosition(e.target.value)}
              >
                <option value="">-- Aucun --</option>
                {POSITIONS.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>
          </div>
          {editPlayerError && <p className="text-red-400 text-xs">{editPlayerError}</p>}
          <div className="flex gap-2 justify-end pt-2">
            <button type="button" onClick={() => setEditingPlayer(null)} className="btn-secondary btn-sm">
              Annuler
            </button>
            <button type="submit" className="btn-primary btn-sm flex items-center gap-1.5">
              <Check size={14} /> Enregistrer
            </button>
          </div>
        </form>
      </Modal>
    </div>
  )
}
