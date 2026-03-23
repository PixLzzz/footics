import { useEffect, useState, useRef, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import {
  getMatch, getEvents, createEvent, deleteEvent,
  analyzeMatch, getAnalysisProgress, getTracks, assignTrack, unassignTrack, markReferee,
  getPlayers, detectEvents, getBallStats, getTrackingAt,
  videoUrl, trackThumbnailUrl,
} from '../api'
import {
  Play, Pause, SkipBack, SkipForward, Scan, Loader,
  Plus, Trash2, CheckCircle, Target, Zap, ArrowRight, ArrowLeft,
  Eye, EyeOff, MousePointer, X, UserCheck, Users,
} from 'lucide-react'

const EVENT_TYPES = [
  { value: 'goal', label: 'But', emoji: '⚽' },
  { value: 'assist', label: 'Passe dé', emoji: '🎯' },
  { value: 'shot', label: 'Tir', emoji: '💥' },
  { value: 'shot_on_target', label: 'Tir cadré', emoji: '🥅' },
  { value: 'pass', label: 'Passe', emoji: '➡️' },
  { value: 'key_pass', label: 'Passe clé', emoji: '🔑' },
  { value: 'interception', label: 'Interception', emoji: '🛡️' },
  { value: 'tackle', label: 'Tacle', emoji: '🦶' },
  { value: 'foul', label: 'Faute', emoji: '🟨' },
  { value: 'save', label: 'Arrêt', emoji: '🧤' },
  { value: 'dribble', label: 'Dribble', emoji: '✨' },
]

function formatTime(s) {
  if (!s && s !== 0) return '--:--'
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

export default function MatchDetail() {
  const { id } = useParams()
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const containerRef = useRef(null)
  const rafRef = useRef(null)
  const lastDrawnTs = useRef(null)

  const [match, setMatch] = useState(null)
  const [events, setEvents] = useState([])
  const [players, setPlayers] = useState([])
  const [tracks, setTracks] = useState([])
  const [currentTime, setCurrentTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [tab, setTab] = useState('events')

  const [selectedPlayer, setSelectedPlayer] = useState('')
  const [selectedEvent, setSelectedEvent] = useState('goal')
  const [progress, setProgress] = useState(null)
  const [homeAttacksRight, setHomeAttacksRight] = useState(true)
  const [detecting, setDetecting] = useState(false)
  const [detectError, setDetectError] = useState('')
  const [ballCount, setBallCount] = useState(null)

  // Overlay state
  const [overlayEnabled, setOverlayEnabled] = useState(true)
  const [assignMode, setAssignMode] = useState(false)
  const [overlayBoxes, setOverlayBoxes] = useState([])
  const [overlayTs, setOverlayTs] = useState(null)
  const [assignPopup, setAssignPopup] = useState(null) // { x, y, trackId, bbox, player }
  const overlayCache = useRef({}) // timestamp -> boxes

  // ── Data loading ─────────────────────────────────────────────────────

  const loadMatch = useCallback(async () => {
    const m = await getMatch(id)
    setMatch(m)
    const ev = await getEvents(id)
    setEvents(ev)
    const p = await getPlayers()
    setPlayers(p)
    if (m.status === 'analyzed') {
      const tr = await getTracks(id, 30)
      setTracks(tr)
      getBallStats(id).then(s => setBallCount(s.ball_detections)).catch(() => {})
    }
  }, [id])

  useEffect(() => { loadMatch() }, [loadMatch])

  // Analysis progress polling
  useEffect(() => {
    if (match?.status !== 'processing') {
      setProgress(null)
      return
    }
    const interval = setInterval(async () => {
      const p = await getAnalysisProgress(id)
      setProgress(p)
      if (p.status !== 'processing') {
        clearInterval(interval)
        const m = await getMatch(id)
        setMatch(m)
        setProgress(null)
        if (m.status === 'analyzed') {
          const tr = await getTracks(id, 30)
          setTracks(tr)
          getBallStats(id).then(s => setBallCount(s.ball_detections)).catch(() => {})
        }
      }
    }, 1500)
    return () => clearInterval(interval)
  }, [match?.status, id])

  // ── Video controls ───────────────────────────────────────────────────

  const handleTimeUpdate = () => {
    if (videoRef.current) setCurrentTime(videoRef.current.currentTime)
  }

  const togglePlay = () => {
    if (!videoRef.current) return
    if (playing) videoRef.current.pause()
    else videoRef.current.play()
    setPlaying(!playing)
  }

  const seek = (seconds) => {
    if (!videoRef.current) return
    videoRef.current.currentTime = Math.max(0, videoRef.current.currentTime + seconds)
  }

  const seekTo = (t, autoPlay = false) => {
    if (!videoRef.current) return
    videoRef.current.currentTime = t
    if (autoPlay && videoRef.current.paused) {
      videoRef.current.play()
    }
  }

  // ── Overlay drawing ──────────────────────────────────────────────────

  const fetchOverlayData = useCallback(async (t) => {
    if (match?.status !== 'analyzed') return
    // Round to nearest 0.1s to reduce API calls
    const rounded = Math.round(t * 10) / 10
    if (overlayCache.current[rounded]) {
      setOverlayBoxes(overlayCache.current[rounded])
      setOverlayTs(rounded)
      return
    }
    try {
      const data = await getTrackingAt(id, rounded)
      overlayCache.current[rounded] = data.boxes || []
      setOverlayBoxes(data.boxes || [])
      setOverlayTs(data.timestamp)
    } catch {
      // silently fail
    }
  }, [id, match?.status])

  const drawOverlay = useCallback(() => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video || !overlayEnabled) return

    const rect = video.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr

    const ctx = canvas.getContext('2d')
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, rect.width, rect.height)

    if (!overlayBoxes.length) return

    const w = rect.width
    const h = rect.height

    for (const box of overlayBoxes) {
      const [bx, by, bw, bh] = box.bbox
      const x = bx * w
      const y = by * h
      const bWidth = bw * w
      const bHeight = bh * h

      if (box.is_referee) {
        // Referee — yellow dashed box
        ctx.strokeStyle = 'rgba(250, 204, 21, 0.6)'
        ctx.lineWidth = 2
        ctx.setLineDash([6, 4])
        ctx.strokeRect(x, y, bWidth, bHeight)
        ctx.setLineDash([])

        // Referee label
        const label = 'Arbitre'
        ctx.font = 'bold 10px system-ui, sans-serif'
        const textW = ctx.measureText(label).width
        const labelH = 16
        const labelY = y - labelH - 2
        ctx.fillStyle = 'rgba(250, 204, 21, 0.8)'
        ctx.beginPath()
        ctx.roundRect(x, Math.max(0, labelY), textW + 8, labelH, 3)
        ctx.fill()
        ctx.fillStyle = '#000'
        ctx.fillText(label, x + 4, Math.max(0, labelY) + 12)
      } else if (box.player) {
        // Assigned player — solid colored box
        const color = box.player.team_color || '#3b82f6'
        ctx.strokeStyle = color
        ctx.lineWidth = 2.5
        ctx.strokeRect(x, y, bWidth, bHeight)

        // Label background
        const label = box.player.jersey ? `#${box.player.jersey} ${box.player.name}` : box.player.name
        ctx.font = 'bold 11px system-ui, sans-serif'
        const textW = ctx.measureText(label).width
        const labelH = 18
        const labelY = y - labelH - 2
        ctx.fillStyle = color
        ctx.beginPath()
        ctx.roundRect(x, Math.max(0, labelY), textW + 10, labelH, 3)
        ctx.fill()

        // Label text
        ctx.fillStyle = '#fff'
        ctx.fillText(label, x + 5, Math.max(0, labelY) + 13)
      } else {
        // Unassigned — dashed gray box
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.5)'
        ctx.lineWidth = 1.5
        ctx.setLineDash([5, 5])
        ctx.strokeRect(x, y, bWidth, bHeight)
        ctx.setLineDash([])

        // Small track ID
        if (assignMode) {
          ctx.fillStyle = 'rgba(148, 163, 184, 0.7)'
          ctx.font = '9px system-ui'
          ctx.fillText(`#${box.track_id}`, x + 2, y + bHeight - 3)
        }
      }
    }

    // Highlight hovered box in assign mode
    if (assignPopup) {
      const [bx, by, bw, bh] = assignPopup.bbox
      ctx.strokeStyle = '#facc15'
      ctx.lineWidth = 3
      ctx.strokeRect(bx * w, by * h, bw * w, bh * h)
    }
  }, [overlayBoxes, overlayEnabled, assignMode, assignPopup])

  // Fetch overlay data when time changes
  useEffect(() => {
    if (!overlayEnabled || match?.status !== 'analyzed') return
    fetchOverlayData(currentTime)
  }, [Math.round(currentTime * 5) / 5, overlayEnabled, match?.status])

  // Redraw overlay when data or state changes
  useEffect(() => {
    drawOverlay()
  }, [drawOverlay])

  // Animation frame loop for smooth overlay during playback
  useEffect(() => {
    if (!playing || !overlayEnabled) return
    let active = true
    const loop = () => {
      if (!active) return
      drawOverlay()
      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
    return () => {
      active = false
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [playing, overlayEnabled, drawOverlay])

  // ── Canvas click handling (assign mode) ──────────────────────────────

  const handleCanvasClick = (e) => {
    if (!assignMode) return

    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const clickX = (e.clientX - rect.left) / rect.width
    const clickY = (e.clientY - rect.top) / rect.height

    // Pause video
    if (videoRef.current && !videoRef.current.paused) {
      videoRef.current.pause()
    }

    // Find clicked box
    let clicked = null
    for (const box of overlayBoxes) {
      const [bx, by, bw, bh] = box.bbox
      if (clickX >= bx && clickX <= bx + bw && clickY >= by && clickY <= by + bh) {
        // Prefer the smallest box (most precise)
        if (!clicked || (bw * bh < clicked.bbox[2] * clicked.bbox[3])) {
          clicked = box
        }
      }
    }

    if (clicked) {
      // Show popup near the click
      setAssignPopup({
        screenX: e.clientX - rect.left,
        screenY: e.clientY - rect.top,
        trackId: clicked.track_id,
        bbox: clicked.bbox,
        player: clicked.player || null,
        isReferee: clicked.is_referee || false,
      })
    } else {
      setAssignPopup(null)
    }
  }

  const handleMarkReferee = async (trackId) => {
    await markReferee(id, trackId)
    overlayCache.current = {}
    setAssignPopup(null)
    const tr = await getTracks(id, 30)
    setTracks(tr)
    fetchOverlayData(currentTime)
  }

  const handleAssignFromOverlay = async (trackId, playerId) => {
    await assignTrack(id, { track_id: trackId, player_id: playerId })
    // Clear cache so we refetch with new assignment data
    overlayCache.current = {}
    setAssignPopup(null)
    // Reload tracks + refresh overlay
    const tr = await getTracks(id, 30)
    setTracks(tr)
    fetchOverlayData(currentTime)
  }

  const handleUnassignFromOverlay = async (trackId) => {
    await unassignTrack(id, trackId)
    overlayCache.current = {}
    setAssignPopup(null)
    const tr = await getTracks(id, 30)
    setTracks(tr)
    fetchOverlayData(currentTime)
  }

  // ── Event handlers ───────────────────────────────────────────────────

  const handleAddEvent = async (e) => {
    e.preventDefault()
    if (!selectedPlayer) return
    await createEvent(id, {
      player_id: selectedPlayer,
      event_type: selectedEvent,
      timestamp_seconds: currentTime,
    })
    const ev = await getEvents(id)
    setEvents(ev)
  }

  const handleDeleteEvent = async (eventId) => {
    await deleteEvent(eventId)
    const ev = await getEvents(id)
    setEvents(ev)
  }

  const handleAnalyze = async () => {
    await analyzeMatch(id)
    setMatch({ ...match, status: 'processing' })
  }

  const handleDetectEvents = async () => {
    setDetecting(true)
    setDetectError('')
    try {
      await detectEvents(id, homeAttacksRight)
      const ev = await getEvents(id)
      setEvents(ev)
      setTab('events')
    } catch (err) {
      setDetectError(err.message)
    } finally {
      setDetecting(false)
    }
  }

  // ── Computed values ──────────────────────────────────────────────────

  const assignedTracks = tracks.filter(t => t.player_id && !t.is_referee)
  const refereeTracks = tracks.filter(t => t.is_referee)
  const assignedCount = assignedTracks.length
  const assignedPlayerIds = new Set(assignedTracks.map(t => t.player_id))

  // ── Render ───────────────────────────────────────────────────────────

  if (!match) return <div className="flex justify-center py-20"><Loader className="animate-spin text-blue-500" /></div>

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">{match.title}</h1>
          <p className="text-gray-500 text-sm">
            {match.team_home && match.team_away
              ? <>{match.team_home.name} vs {match.team_away.name} &middot; </>
              : null}
            {formatTime(match.duration_seconds)}
          </p>
        </div>
        {match.status === 'processing' ? (
          <div className="flex items-center gap-3 min-w-[280px]">
            <Loader size={16} className="animate-spin text-yellow-400 shrink-0" />
            <div className="flex-1">
              <div className="flex justify-between text-xs mb-1">
                <span className="text-yellow-400">Analyse en cours...</span>
                <span className="text-yellow-300 font-mono">{progress ? `${progress.percent}%` : '0%'}</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-blue-500 to-yellow-400 h-full rounded-full transition-all duration-500"
                  style={{ width: `${progress?.percent || 0}%` }}
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            {match.status === 'analyzed' && (
              <span className="flex items-center gap-2 text-emerald-400 text-sm mr-2">
                <CheckCircle size={16} /> Analysé
              </span>
            )}
            {match.status === 'error' && (
              <span className="text-red-400 text-sm mr-2">Erreur</span>
            )}
            <button onClick={handleAnalyze} className="btn-primary flex items-center gap-2">
              <Scan size={16} /> {match.status === 'uploaded' ? "Lancer l'analyse YOLO" : "Relancer l'analyse"}
            </button>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* ── Video Player with Overlay ──────────────────────────── */}
        <div className="xl:col-span-2">
          <div className="card !p-0 overflow-hidden">
            <div ref={containerRef} className="relative bg-black">
              <video
                ref={videoRef}
                src={videoUrl(id)}
                className="w-full block"
                preload="auto"
                muted
                onTimeUpdate={handleTimeUpdate}
                onPlay={() => setPlaying(true)}
                onPause={() => { setPlaying(false); fetchOverlayData(videoRef.current?.currentTime || 0) }}
                onSeeked={() => fetchOverlayData(videoRef.current?.currentTime || 0)}
              />
              {/* Overlay canvas */}
              {match.status === 'analyzed' && overlayEnabled && (
                <canvas
                  ref={canvasRef}
                  className={`absolute inset-0 w-full h-full ${
                    assignMode ? 'cursor-crosshair' : 'pointer-events-none'
                  }`}
                  onClick={handleCanvasClick}
                />
              )}
              {/* Assignment popup */}
              {assignPopup && (
                <div
                  className="absolute z-20 animate-in"
                  style={{
                    left: Math.min(assignPopup.screenX, (containerRef.current?.clientWidth || 400) - 220),
                    top: Math.min(assignPopup.screenY + 10, (containerRef.current?.clientHeight || 300) - 180),
                  }}
                >
                  <div className="bg-slate-900/95 backdrop-blur-md border border-blue-500/50 rounded-xl shadow-2xl p-3 w-52">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-medium text-gray-400">
                        Track #{assignPopup.trackId}
                      </span>
                      <button
                        onClick={() => setAssignPopup(null)}
                        className="text-gray-500 hover:text-white p-0.5"
                      >
                        <X size={12} />
                      </button>
                    </div>
                    {assignPopup.isReferee ? (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 px-2 py-1.5 bg-yellow-900/30 border border-yellow-500/30 rounded-lg">
                          <span className="text-sm">🟨</span>
                          <span className="text-sm font-medium text-yellow-300">Arbitre</span>
                        </div>
                        <button
                          onClick={() => handleUnassignFromOverlay(assignPopup.trackId)}
                          className="w-full text-xs py-1.5 text-gray-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
                        >
                          Retirer le marquage
                        </button>
                      </div>
                    ) : assignPopup.player ? (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 px-2 py-1.5 bg-slate-800 rounded-lg">
                          <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: assignPopup.player.team_color }} />
                          <span className="text-sm font-medium">
                            {assignPopup.player.jersey && `#${assignPopup.player.jersey} `}
                            {assignPopup.player.name}
                          </span>
                        </div>
                        <div className="flex gap-1.5">
                          <select
                            className="input text-xs py-1 flex-1"
                            defaultValue={assignPopup.player.player_id}
                            onChange={(e) => {
                              if (e.target.value) handleAssignFromOverlay(assignPopup.trackId, e.target.value)
                            }}
                          >
                            {players.map((p) => (
                              <option key={p.id} value={p.id}>
                                {p.jersey_number ? `#${p.jersey_number} ` : ''}{p.name}
                              </option>
                            ))}
                          </select>
                          <button
                            onClick={() => handleUnassignFromOverlay(assignPopup.trackId)}
                            className="px-2 py-1 text-xs text-red-400 hover:bg-red-900/20 rounded-lg transition-colors"
                          >
                            <Trash2 size={12} />
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <select
                          className="input text-xs py-1.5 w-full"
                          defaultValue=""
                          onChange={(e) => {
                            if (e.target.value) handleAssignFromOverlay(assignPopup.trackId, e.target.value)
                          }}
                          autoFocus
                        >
                          <option value="">-- Qui est ce joueur ? --</option>
                          {players.map((p) => (
                            <option key={p.id} value={p.id}>
                              {p.jersey_number ? `#${p.jersey_number} ` : ''}{p.name} ({p.team_name})
                            </option>
                          ))}
                        </select>
                        <button
                          onClick={() => handleMarkReferee(assignPopup.trackId)}
                          className="w-full flex items-center justify-center gap-1.5 py-1.5 text-xs font-medium text-yellow-400 hover:bg-yellow-500/10 border border-yellow-500/30 rounded-lg transition-colors"
                        >
                          🟨 C'est l'arbitre
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}
              {/* Assign mode indicator */}
              {assignMode && (
                <div className="absolute top-3 left-3 flex items-center gap-2 px-3 py-1.5 bg-yellow-500/20 backdrop-blur-sm border border-yellow-500/40 rounded-lg">
                  <MousePointer size={13} className="text-yellow-400" />
                  <span className="text-xs font-medium text-yellow-300">Cliquez sur un joueur pour l'identifier</span>
                </div>
              )}
            </div>

            {/* Controls bar */}
            <div className="flex items-center gap-3 px-4 py-3 bg-slate-900/80 border-t border-blue-900/20">
              <button onClick={() => seek(-10)} className="text-gray-500 hover:text-white transition-colors">
                <SkipBack size={16} />
              </button>
              <button onClick={togglePlay} className="text-white hover:text-blue-400 transition-colors">
                {playing ? <Pause size={20} /> : <Play size={20} />}
              </button>
              <button onClick={() => seek(10)} className="text-gray-500 hover:text-white transition-colors">
                <SkipForward size={16} />
              </button>
              <span className="text-xs font-mono text-gray-500 ml-1">
                {formatTime(currentTime)} / {formatTime(match.duration_seconds)}
              </span>

              <div className="flex-1" />

              {/* Overlay controls */}
              {match.status === 'analyzed' && (
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => { setOverlayEnabled(!overlayEnabled); setAssignPopup(null) }}
                    className={`p-1.5 rounded-lg text-xs transition-colors ${
                      overlayEnabled ? 'text-blue-400 bg-blue-600/20' : 'text-gray-600 hover:text-gray-400'
                    }`}
                    title="Afficher/masquer les bounding boxes"
                  >
                    {overlayEnabled ? <Eye size={15} /> : <EyeOff size={15} />}
                  </button>
                  <button
                    onClick={() => { setAssignMode(!assignMode); setAssignPopup(null) }}
                    className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      assignMode
                        ? 'text-yellow-300 bg-yellow-500/20 border border-yellow-500/40'
                        : 'text-gray-500 hover:text-gray-300 hover:bg-slate-800'
                    }`}
                    title="Mode assignation : cliquez sur les joueurs"
                  >
                    <MousePointer size={13} />
                    {assignMode ? 'Assignation ON' : 'Assigner'}
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Add Event Form */}
          <div className="card mt-4">
            <h3 className="font-semibold mb-3 flex items-center gap-2 text-sm">
              <Target size={15} className="text-red-500" /> Ajouter un événement à {formatTime(currentTime)}
            </h3>
            <form onSubmit={handleAddEvent} className="flex gap-2 flex-wrap">
              <select
                className="input w-auto flex-1"
                value={selectedPlayer}
                onChange={(e) => setSelectedPlayer(e.target.value)}
              >
                <option value="">-- Joueur --</option>
                {players.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.jersey_number ? `#${p.jersey_number} ` : ''}{p.name} ({p.team_name})
                  </option>
                ))}
              </select>
              <select
                className="input w-auto"
                value={selectedEvent}
                onChange={(e) => setSelectedEvent(e.target.value)}
              >
                {EVENT_TYPES.map((et) => (
                  <option key={et.value} value={et.value}>{et.emoji} {et.label}</option>
                ))}
              </select>
              <button type="submit" className="btn-primary btn-sm flex items-center gap-1" disabled={!selectedPlayer}>
                <Plus size={14} /> Ajouter
              </button>
            </form>
          </div>
        </div>

        {/* ── Side Panel ────────────────────────────────────────── */}
        <div className="card">
          <div className="flex border-b border-blue-900/30 mb-4 -mx-6 -mt-6 px-6">
            <button
              className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                tab === 'events' ? 'border-blue-500 text-blue-400' : 'border-transparent text-gray-500 hover:text-gray-300'
              }`}
              onClick={() => setTab('events')}
            >
              Events ({events.length})
            </button>
            <button
              className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                tab === 'players' ? 'border-blue-500 text-blue-400' : 'border-transparent text-gray-500 hover:text-gray-300'
              }`}
              onClick={() => setTab('players')}
            >
              Joueurs ({assignedCount})
            </button>
            <button
              className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                tab === 'detect' ? 'border-yellow-500 text-yellow-400' : 'border-transparent text-gray-500 hover:text-gray-300'
              }`}
              onClick={() => setTab('detect')}
            >
              Auto-detect
            </button>
          </div>

          {/* ── Events tab ──────────────────────────────────────── */}
          {tab === 'events' && (
            <div className="space-y-1.5 max-h-[600px] overflow-y-auto pr-1">
              {events.length === 0 ? (
                <p className="text-gray-500 text-sm text-center py-8">
                  Aucun événement enregistré.
                </p>
              ) : events.map((ev) => (
                <div
                  key={ev.id}
                  className="flex items-center gap-2 px-3 py-2 bg-slate-800/60 hover:bg-slate-800 rounded-xl text-sm cursor-pointer transition-colors"
                  onClick={() => seekTo(ev.event_type === 'goal' ? Math.max(0, ev.timestamp_seconds - 5) : ev.timestamp_seconds, true)}
                >
                  <span className="font-mono text-[10px] text-gray-500 w-10 shrink-0">{formatTime(ev.timestamp_seconds)}</span>
                  <span className="shrink-0">{EVENT_TYPES.find((e) => e.value === ev.event_type)?.emoji}</span>
                  <div className="flex-1 min-w-0">
                    <span className="font-medium">{ev.player_name}</span>
                    <span className="text-gray-500 ml-1 text-xs">
                      {EVENT_TYPES.find((e) => e.value === ev.event_type)?.label}
                    </span>
                  </div>
                  <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: ev.team_color }} />
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDeleteEvent(ev.id) }}
                    className="text-gray-600 hover:text-red-400 shrink-0"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* ── Players / Assignment tab ────────────────────────── */}
          {tab === 'players' && (
            <div className="space-y-4">
              {match.status !== 'analyzed' ? (
                <p className="text-gray-500 text-sm text-center py-8">
                  Lancez d'abord l'analyse YOLO pour détecter les joueurs.
                </p>
              ) : (
                <>
                  <div className="flex items-center gap-3 p-3 bg-slate-800/60 rounded-xl">
                    <UserCheck size={18} className="text-blue-400 shrink-0" />
                    <div className="flex-1">
                      <p className="text-sm font-medium">
                        {assignedCount} joueur{assignedCount !== 1 && 's'} identifié{assignedCount !== 1 && 's'}
                      </p>
                      <p className="text-xs text-gray-500">
                        {assignedCount === 0
                          ? "Activez le mode 'Assigner' et cliquez sur les joueurs dans la vidéo"
                          : `sur ${tracks.length} tracks détectés`}
                      </p>
                    </div>
                  </div>

                  {!assignMode && assignedCount === 0 && (
                    <button
                      onClick={() => setAssignMode(true)}
                      className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-yellow-500/10 hover:bg-yellow-500/20 border border-yellow-500/30 rounded-xl text-yellow-300 text-sm font-medium transition-colors"
                    >
                      <MousePointer size={15} /> Commencer l'identification des joueurs
                    </button>
                  )}

                  {/* List of assigned players */}
                  {assignedCount > 0 && (
                    <div className="space-y-1.5 max-h-[400px] overflow-y-auto pr-1">
                      {assignedTracks.map((t) => {
                        const player = players.find(p => p.id === t.player_id)
                        if (!player) return null
                        return (
                          <div
                            key={t.track_id}
                            className="flex items-center gap-3 px-3 py-2.5 bg-slate-800/60 rounded-xl text-sm"
                          >
                            <div
                              className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shrink-0"
                              style={{ backgroundColor: player.team_color + '25', color: player.team_color }}
                            >
                              {player.jersey_number ?? '?'}
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="font-medium truncate">{player.name}</p>
                              <p className="text-[10px] text-gray-500">
                                Track #{t.track_id} &middot; {t.frame_count} frames &middot; {formatTime(t.first_seen)}→{formatTime(t.last_seen)}
                              </p>
                            </div>
                            <button
                              onClick={() => handleUnassignFromOverlay(t.track_id)}
                              className="p-1 text-gray-600 hover:text-red-400 transition-colors"
                            >
                              <X size={13} />
                            </button>
                          </div>
                        )
                      })}
                    </div>
                  )}

                  {/* Referee tracks */}
                  {refereeTracks.length > 0 && (
                    <div>
                      <p className="text-xs text-gray-600 mb-1.5">Arbitre{refereeTracks.length > 1 && 's'} :</p>
                      <div className="space-y-1">
                        {refereeTracks.map((t) => (
                          <div key={t.track_id} className="flex items-center gap-2 px-3 py-2 bg-yellow-900/10 border border-yellow-500/20 rounded-lg text-xs">
                            <span className="text-sm">🟨</span>
                            <span className="text-yellow-300 font-medium flex-1">Arbitre</span>
                            <span className="text-gray-600 text-[10px]">Track #{t.track_id}</span>
                            <button
                              onClick={() => handleUnassignFromOverlay(t.track_id)}
                              className="p-0.5 text-gray-600 hover:text-red-400 transition-colors"
                            >
                              <X size={11} />
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Unassigned players hint */}
                  {players.length > 0 && (
                    <div>
                      <p className="text-xs text-gray-600 mb-2">Joueurs non identifiés :</p>
                      <div className="flex flex-wrap gap-1.5">
                        {players.filter(p => !assignedPlayerIds.has(p.id)).map((p) => (
                          <span
                            key={p.id}
                            className="px-2 py-1 bg-slate-800/40 rounded-lg text-[10px] text-gray-500 flex items-center gap-1"
                          >
                            <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: p.team_color }} />
                            {p.jersey_number ? `#${p.jersey_number} ` : ''}{p.name}
                          </span>
                        ))}
                        {players.filter(p => !assignedPlayerIds.has(p.id)).length === 0 && (
                          <span className="text-[10px] text-emerald-500">Tous les joueurs sont identifiés !</span>
                        )}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* ── Auto-detect tab ─────────────────────────────────── */}
          {tab === 'detect' && (
            <div className="space-y-4">
              {match.status !== 'analyzed' ? (
                <p className="text-gray-500 text-sm text-center py-8">
                  Lancez d'abord l'analyse YOLO pour détecter joueurs et ballon.
                </p>
              ) : (
                <>
                  {ballCount !== null && (
                    <div className="text-xs text-gray-500 mb-2">
                      Ballon détecté dans {ballCount} frames
                      {ballCount === 0 && (
                        <span className="text-red-400 ml-1">
                          — le ballon n'a pas été détecté, la détection sera limitée
                        </span>
                      )}
                    </div>
                  )}

                  <div>
                    <p className="text-sm font-medium mb-2">Direction d'attaque</p>
                    <div className="flex gap-2">
                      <button
                        onClick={() => setHomeAttacksRight(true)}
                        className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-sm transition-all ${
                          homeAttacksRight
                            ? 'bg-blue-600/20 border border-blue-500/40 text-blue-300'
                            : 'bg-slate-800/60 text-gray-500 hover:bg-slate-800'
                        }`}
                      >
                        {match.team_home?.name || 'Domicile'} <ArrowRight size={14} />
                      </button>
                      <button
                        onClick={() => setHomeAttacksRight(false)}
                        className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-sm transition-all ${
                          !homeAttacksRight
                            ? 'bg-blue-600/20 border border-blue-500/40 text-blue-300'
                            : 'bg-slate-800/60 text-gray-500 hover:bg-slate-800'
                        }`}
                      >
                        <ArrowLeft size={14} /> {match.team_home?.name || 'Domicile'}
                      </button>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 p-3 bg-slate-800/60 rounded-xl">
                    <Users size={16} className={assignedCount > 0 ? 'text-emerald-400' : 'text-red-400'} />
                    <div>
                      <p className="text-sm">
                        {assignedCount > 0 ? (
                          <span className="text-emerald-400">{assignedCount} joueurs assignés</span>
                        ) : (
                          <span className="text-red-400">Aucun joueur assigné</span>
                        )}
                      </p>
                      {assignedCount === 0 && (
                        <p className="text-[10px] text-gray-500">
                          Identifiez d'abord les joueurs dans l'onglet Joueurs
                        </p>
                      )}
                    </div>
                  </div>

                  {detectError && (
                    <p className="text-red-400 text-xs bg-red-900/10 border border-red-900/30 rounded-lg px-3 py-2">{detectError}</p>
                  )}

                  <button
                    onClick={handleDetectEvents}
                    disabled={detecting || assignedCount === 0}
                    className="btn-primary w-full flex items-center justify-center gap-2"
                  >
                    {detecting ? (
                      <><Loader size={16} className="animate-spin" /> Détection en cours...</>
                    ) : (
                      <><Zap size={16} /> Détecter les événements</>
                    )}
                  </button>

                  <p className="text-[10px] text-gray-600">
                    Remplace tous les événements existants par les événements auto-détectés
                    (passes, interceptions, tirs, buts, assists, dribbles, key passes, saves).
                  </p>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
