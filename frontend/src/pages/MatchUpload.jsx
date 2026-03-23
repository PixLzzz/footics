import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadMatch, getTeams } from '../api'
import { Upload, Film, Loader } from 'lucide-react'

export default function MatchUpload() {
  const navigate = useNavigate()
  const [teams, setTeams] = useState([])
  const [title, setTitle] = useState('')
  const [teamId, setTeamId] = useState('')
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => { getTeams().then(setTeams) }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file || !title) return

    setUploading(true)
    setError('')

    const fd = new FormData()
    fd.append('title', title)
    fd.append('video', file)
    if (teamId) fd.append('team_id', teamId)

    try {
      const result = await uploadMatch(fd)
      navigate(`/match/${result.id}`)
    } catch (err) {
      setError(err.message)
      setUploading(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const f = e.dataTransfer?.files?.[0]
    if (f && f.type.startsWith('video/')) setFile(f)
  }

  return (
    <div className="max-w-2xl">
      <div className="mb-8">
        <h1 className="text-2xl font-bold">Uploader un match</h1>
        <p className="text-sm text-gray-500 mt-1">Ajoutez une vidéo pour commencer l'analyse</p>
      </div>

      <form onSubmit={handleSubmit} className="card space-y-6">
        <div>
          <label className="label">Titre du match *</label>
          <input
            type="text"
            className="input"
            placeholder="Ex: Match du samedi 15 mars"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            required
          />
        </div>

        <div>
          <label className="label">Equipe analysée</label>
          <select
            className="input"
            value={teamId}
            onChange={(e) => setTeamId(e.target.value)}
          >
            <option value="">-- Aucune --</option>
            {teams.map((t) => (
              <option key={t.id} value={t.id}>{t.name}</option>
            ))}
          </select>
          <p className="text-xs text-gray-600 mt-1">
            Seuls les joueurs de cette équipe seront trackés et analysés
          </p>
        </div>

        <div>
          <label className="label">Vidéo du match *</label>
          <label
            className={`flex flex-col items-center justify-center w-full h-44 border-2 border-dashed rounded-2xl cursor-pointer transition-all ${
              file
                ? 'border-blue-500/50 bg-blue-950/20'
                : 'border-blue-900/40 hover:border-blue-500/50 hover:bg-blue-950/10'
            }`}
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
          >
            {file ? (
              <div className="text-center">
                <Film size={36} className="mx-auto mb-3 text-blue-400" />
                <p className="text-sm font-medium text-gray-200">{file.name}</p>
                <p className="text-xs text-gray-500 mt-1">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
                <p className="text-xs text-blue-500 mt-2 hover:underline">Changer de fichier</p>
              </div>
            ) : (
              <div className="text-center">
                <Upload size={36} className="mx-auto mb-3 text-blue-900" />
                <p className="text-sm text-gray-400">Cliquez ou glissez une vidéo</p>
                <p className="text-xs text-gray-600 mt-1">MP4, AVI, MOV, MKV, WebM</p>
              </div>
            )}
            <input
              type="file"
              className="hidden"
              accept="video/*"
              onChange={(e) => setFile(e.target.files[0])}
            />
          </label>
        </div>

        {error && (
          <p className="text-sm text-red-400 bg-red-900/10 border border-red-900/30 rounded-lg px-3 py-2">{error}</p>
        )}

        <button
          type="submit"
          disabled={!file || !title || uploading}
          className="btn-primary w-full flex items-center justify-center gap-2"
        >
          {uploading ? (
            <><Loader size={16} className="animate-spin" /> Upload en cours...</>
          ) : (
            <><Upload size={16} /> Uploader le match</>
          )}
        </button>
      </form>
    </div>
  )
}
