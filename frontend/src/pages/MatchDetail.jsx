import { useEffect, useState, useRef, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import {
  getMatch, getEvents, createEvent, deleteEvent,
  analyzeMatch, cancelAnalysis, getAnalysisProgress, getTracks, assignTrack, unassignTrack, unassignAllTracks, markReferee,
  getPlayers, getTrackingBulk,
  runFullPipeline, getPipelineProgress,
  videoUrl, getTrackConfidence, exportMatchData,
} from '../api'
import {
  Play, Pause, SkipBack, SkipForward, Scan, Loader,
  Plus, Trash2, CheckCircle, Target, Zap,
  Eye, EyeOff, MousePointer, X, UserCheck, Rocket, Download,
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

  const [match, setMatch] = useState(null)
  const [events, setEvents] = useState([])
  const [players, setPlayers] = useState([])
  const [tracks, setTracks] = useState([])
  const [currentTime, setCurrentTime] = useState(0)
  const [playing, setPlaying] = useState(false)

  const [selectedPlayer, setSelectedPlayer] = useState('')
  const [selectedEvent, setSelectedEvent] = useState('goal')
  const [progress, setProgress] = useState(null)
  // Full pipeline state
  const [pipelineRunning, setPipelineRunning] = useState(false)
  const [pipelineProgress, setPipelineProgress] = useState(null)
  const [pipelineResult, setPipelineResult] = useState(null)

  const [trackConfidence, setTrackConfidence] = useState({}) // track_id → score

  // Overlay state
  const [overlayEnabled, setOverlayEnabled] = useState(true)
  const [assignMode, setAssignMode] = useState(false)
  const [overlayBoxes, setOverlayBoxes] = useState([]) // interpolated boxes for drawing
  const [assignPopup, setAssignPopup] = useState(null) // { x, y, trackId, bbox, player }

  // Bulk tracking cache: pre-fetched chunks of ~10s
  // Each chunk: { timestamps: number[], frames: {[ts]: boxes[]}, assignments: {}, referee_tracks: [], start, end }
  const bulkCache = useRef([]) // array of loaded chunks
  const pendingChunks = useRef(new Set()) // chunks currently being fetched (keyed by start time)

  // ── Data loading ─────────────────────────────────────────────────────

  const loadMatch = useCallback(async () => {
    const m = await getMatch(id)
    setMatch(m)
    const ev = await getEvents(id)
    setEvents(ev)
    const allPlayers = await getPlayers()
    // Filter to only the match's team if set
    setPlayers(m.team ? allPlayers.filter(p => p.team_id === m.team.id) : allPlayers)
    if (m.status === 'analyzed') {
      const tr = await getTracks(id, 30)
      setTracks(tr)
      getTrackConfidence(id).then(r => setTrackConfidence(r.scores || {})).catch(() => {})
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

  // ── Overlay: bulk pre-fetch + client-side interpolation ─────────────

  const CHUNK_SIZE = 10 // seconds per chunk

  // Find which chunk covers a given time
  const findChunk = useCallback((t) => {
    return bulkCache.current.find(c => t >= c.start && t <= c.end)
  }, [])

  // Binary search for the two surrounding timestamps in a sorted array
  const findSurrounding = useCallback((timestamps, t) => {
    if (!timestamps.length) return [null, null]
    let lo = 0, hi = timestamps.length - 1
    // If t is before first or after last
    if (t <= timestamps[0]) return [timestamps[0], timestamps.length > 1 ? timestamps[1] : null]
    if (t >= timestamps[hi]) return [timestamps[hi], null]
    // Binary search
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      if (timestamps[mid] <= t) lo = mid + 1
      else hi = mid - 1
    }
    // hi is the last index <= t, lo is the first index > t
    return [timestamps[hi], timestamps[lo] ?? null]
  }, [])

  // Interpolate boxes between two timestamps using cached chunk data
  const getInterpolatedBoxes = useCallback((chunk, t) => {
    if (!chunk || !chunk.timestamps.length) return []
    const [tsBefore, tsAfter] = findSurrounding(chunk.timestamps, t)
    if (tsBefore === null) return []

    const beforeBoxes = chunk.frames[tsBefore] || []
    const { assignments, referee_tracks, team_labels } = chunk

    // Annotate boxes with player/referee/team info
    const annotate = (box) => {
      const result = { ...box }
      if (referee_tracks.includes(box.track_id)) {
        result.is_referee = true
      } else if (assignments[String(box.track_id)]) {
        result.player = assignments[String(box.track_id)]
      }
      // Attach team label if available
      if (team_labels && team_labels[String(box.track_id)] !== undefined) {
        result.team_label = team_labels[String(box.track_id)]
      }
      return result
    }

    if (tsAfter === null || tsAfter === tsBefore) {
      return beforeBoxes.map(annotate)
    }

    const afterBoxes = chunk.frames[tsAfter] || []
    const dt = tsAfter - tsBefore
    const ratio = dt > 0 ? Math.max(0, Math.min(1, (t - tsBefore) / dt)) : 0

    // Build lookup for after frame
    const afterMap = {}
    for (const box of afterBoxes) {
      afterMap[box.track_id] = box
    }

    return beforeBoxes.map(box => {
      const after = afterMap[box.track_id]
      const annotated = annotate(box)
      if (!after || ratio <= 0) return annotated

      const [bx1, by1, bw1, bh1] = box.bbox
      const [bx2, by2, bw2, bh2] = after.bbox
      return {
        ...annotated,
        bbox: [
          bx1 + (bx2 - bx1) * ratio,
          by1 + (by2 - by1) * ratio,
          bw1 + (bw2 - bw1) * ratio,
          bh1 + (bh2 - bh1) * ratio,
        ],
      }
    })
  }, [findSurrounding])

  // Pre-fetch a chunk of tracking data
  const fetchChunk = useCallback(async (chunkStart) => {
    if (match?.status !== 'analyzed') return
    const key = chunkStart
    if (pendingChunks.current.has(key)) return
    // Already cached?
    if (bulkCache.current.find(c => c.start === chunkStart)) return

    pendingChunks.current.add(key)
    try {
      const chunkEnd = chunkStart + CHUNK_SIZE
      const data = await getTrackingBulk(id, chunkStart, chunkEnd)
      // Normalize frame keys: JSON "0.0" stays as string but JS parses
      // timestamp 0.0 to number 0 — so frames["0.0"] != frames[0].
      // Re-key frames dict using parseFloat so lookups by number work.
      const normalizedFrames = {}
      for (const [k, v] of Object.entries(data.frames)) {
        normalizedFrames[parseFloat(k)] = v
      }
      data.frames = normalizedFrames

      // Avoid duplicates
      if (!bulkCache.current.find(c => c.start === chunkStart)) {
        bulkCache.current.push(data)
        // Keep cache bounded — max 6 chunks (~60s)
        if (bulkCache.current.length > 6) {
          bulkCache.current.shift()
        }
      }
      // After loading, immediately update overlay for the current video time
      const t = videoRef.current?.currentTime ?? 0
      if (t >= chunkStart && t <= chunkStart + CHUNK_SIZE) {
        const boxes = getInterpolatedBoxes(data, t)
        setOverlayBoxes(boxes)
      }
    } catch {
      // silently fail
    } finally {
      pendingChunks.current.delete(key)
    }
  }, [id, match?.status, getInterpolatedBoxes])

  // Get boxes for a given time, triggering fetch if needed
  const updateOverlay = useCallback((t) => {
    if (match?.status !== 'analyzed' || !overlayEnabled) return

    const chunkStart = Math.floor(t / CHUNK_SIZE) * CHUNK_SIZE
    let chunk = findChunk(t)

    if (!chunk) {
      // Trigger fetch for current chunk
      fetchChunk(chunkStart)
      return
    }

    // Pre-fetch next chunk if we're within 2s of the end
    if (t > chunk.end - 2) {
      fetchChunk(chunkStart + CHUNK_SIZE)
    }

    const boxes = getInterpolatedBoxes(chunk, t)
    setOverlayBoxes(boxes)
  }, [match?.status, overlayEnabled, findChunk, fetchChunk, getInterpolatedBoxes])

  // Invalidate bulk cache and refetch current chunk (after assignment changes)
  const invalidateOverlayCache = useCallback(async () => {
    bulkCache.current = []
    pendingChunks.current.clear()
    // Immediately refetch the current chunk so boxes reappear
    if (match?.status === 'analyzed' && videoRef.current) {
      const t = videoRef.current.currentTime || 0
      const chunkStart = Math.floor(t / CHUNK_SIZE) * CHUNK_SIZE
      try {
        const data = await getTrackingBulk(id, chunkStart, chunkStart + CHUNK_SIZE)
        // Normalize frame keys (same "0.0" vs 0 fix)
        const normalizedFrames = {}
        for (const [k, v] of Object.entries(data.frames)) {
          normalizedFrames[parseFloat(k)] = v
        }
        data.frames = normalizedFrames
        bulkCache.current.push(data)
        // Re-compute boxes from the fresh data
        const boxes = getInterpolatedBoxes(data, t)
        setOverlayBoxes(boxes)
      } catch { /* ignore */ }
    }
  }, [id, match?.status, getInterpolatedBoxes])

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

    const w = rect.width
    const h = rect.height

    if (!overlayBoxes.length) return

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
        // Unassigned — color by team label if available
        const TEAM_COLORS = ['rgba(59, 130, 246, 0.6)', 'rgba(239, 68, 68, 0.6)', 'rgba(250, 204, 21, 0.5)']
        const teamColor = box.team_label !== undefined ? TEAM_COLORS[box.team_label] || TEAM_COLORS[0] : 'rgba(148, 163, 184, 0.5)'

        ctx.strokeStyle = teamColor
        ctx.lineWidth = 1.5
        ctx.setLineDash([5, 5])
        ctx.strokeRect(x, y, bWidth, bHeight)
        ctx.setLineDash([])

        // Team label indicator + track ID
        if (assignMode) {
          const labelText = box.team_label !== undefined ? `T${box.team_label} #${box.track_id}` : `#${box.track_id}`
          ctx.fillStyle = teamColor
          ctx.font = '9px system-ui'
          ctx.fillText(labelText, x + 2, y + bHeight - 3)
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

  // Pre-fetch initial chunk when analysis is ready
  useEffect(() => {
    if (match?.status !== 'analyzed' || !overlayEnabled) return
    invalidateOverlayCache()
    fetchChunk(0)
  }, [match?.status, overlayEnabled])

  // Update overlay when time changes (paused or scrubbing)
  useEffect(() => {
    if (!overlayEnabled || match?.status !== 'analyzed') return
    updateOverlay(currentTime)
  }, [Math.round(currentTime * 20) / 20, overlayEnabled, match?.status])

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
      if (videoRef.current) {
        updateOverlay(videoRef.current.currentTime)
      }
      drawOverlay()
      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
    return () => {
      active = false
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [playing, overlayEnabled, drawOverlay, updateOverlay])

  // ── Canvas click handling (assign mode) ──────────────────────────────

  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const clickX = (e.clientX - rect.left) / rect.width
    const clickY = (e.clientY - rect.top) / rect.height

    // Find clicked box
    let clicked = null
    for (const box of overlayBoxes) {
      const [bx, by, bw, bh] = box.bbox
      if (clickX >= bx && clickX <= bx + bw && clickY >= by && clickY <= by + bh) {
        if (!clicked || (bw * bh < clicked.bbox[2] * clicked.bbox[3])) {
          clicked = box
        }
      }
    }

    if (!clicked) {
      setAssignPopup(null)
      return
    }

    // Always allow clicking on assigned players (to re-assign)
    // For unassigned boxes, require assign mode
    if (!assignMode && !clicked.player && !clicked.is_referee) {
      setAssignPopup(null)
      return
    }

    // Pause video
    if (videoRef.current && !videoRef.current.paused) {
      videoRef.current.pause()
    }

    setAssignPopup({
      screenX: e.clientX - rect.left,
      screenY: e.clientY - rect.top,
      trackId: clicked.track_id,
      bbox: clicked.bbox,
      player: clicked.player || null,
      isReferee: clicked.is_referee || false,
    })
  }

  const handleMarkReferee = async (trackId) => {
    await markReferee(id, trackId)
    setAssignPopup(null)
    const tr = await getTracks(id, 30)
    setTracks(tr)
    await invalidateOverlayCache()
  }

  const handleAssignFromOverlay = async (trackId, playerId) => {
    await assignTrack(id, { track_id: trackId, player_id: playerId })
    setAssignPopup(null)
    const tr = await getTracks(id, 30)
    setTracks(tr)
    await invalidateOverlayCache()
  }

  const handleUnassignFromOverlay = async (trackId) => {
    await unassignTrack(id, trackId)
    setAssignPopup(null)
    const tr = await getTracks(id, 30)
    setTracks(tr)
    await invalidateOverlayCache()
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
    try {
      await analyzeMatch(id)
      setMatch({ ...match, status: 'processing' })
    } catch (err) {
      alert('Erreur: ' + err.message)
    }
  }

  const handleCancelAnalysis = async () => {
    try {
      await cancelAnalysis(id)
    } catch { /* ignore */ }
  }

  const handleResetAssignments = async () => {
    if (!confirm('Supprimer toutes les assignations de joueurs ?')) return
    try {
      await unassignAllTracks(id)
      setPipelineResult(null)
      const tr = await getTracks(id, 30)
      setTracks(tr)
      bulkCache.current = []
    } catch (err) {
      alert(err.message)
    }
  }

  // ── Full pipeline (auto-assign + correction + event detection) ──
  const handleRunPipeline = async () => {
    setPipelineRunning(true)
    setPipelineProgress(null)
    setPipelineResult(null)
    try {
      await runFullPipeline(id)
    } catch (err) {
      setPipelineResult({ error: err.message })
      setPipelineRunning(false)
    }
  }

  // Pipeline progress polling
  useEffect(() => {
    if (!pipelineRunning) return
    const interval = setInterval(async () => {
      try {
        const p = await getPipelineProgress(id)
        setPipelineProgress(p)
        if (p.done || p.percent >= 100) {
          clearInterval(interval)
          if (p.error) {
            setPipelineResult({ error: p.error })
          } else {
            setPipelineResult({
              assigned: p.assigned || 0,
              corrected: p.corrected || 0,
              events: p.events || 0,
            })
          }
          const tr = await getTracks(id, 30)
          setTracks(tr)
          const ev = await getEvents(id)
          setEvents(ev)
          bulkCache.current = []
          invalidateOverlayCache()
          setPipelineRunning(false)
        }
      } catch { /* ignore */ }
    }, 1000)
    return () => clearInterval(interval)
  }, [pipelineRunning, id])

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
            {match.team
              ? <><span style={{ color: match.team.color }}>{match.team.name}</span> &middot; </>
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
            <button
              onClick={handleCancelAnalysis}
              className="shrink-0 p-1.5 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
              title="Annuler l'analyse"
            >
              <X size={16} />
            </button>
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
                onPause={() => { setPlaying(false); updateOverlay(videoRef.current?.currentTime || 0) }}
                onSeeked={() => updateOverlay(videoRef.current?.currentTime || 0)}
              />
              {/* Overlay canvas */}
              {match.status === 'analyzed' && overlayEnabled && (
                <canvas
                  ref={canvasRef}
                  className={`absolute inset-0 w-full h-full ${assignMode ? 'cursor-crosshair' : 'cursor-pointer'}`}
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
                              {p.jersey_number ? `#${p.jersey_number} ` : ''}{p.name}
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

          {/* Add Event Form (compact) */}
          {match.status === 'analyzed' && (
            <div className="card mt-4">
              <form onSubmit={handleAddEvent} className="flex gap-2 items-center flex-wrap">
                <span className="text-xs text-gray-500 font-mono shrink-0">{formatTime(currentTime)}</span>
                <select
                  className="input w-auto flex-1 text-sm py-1.5"
                  value={selectedPlayer}
                  onChange={(e) => setSelectedPlayer(e.target.value)}
                >
                  <option value="">Joueur...</option>
                  {players.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.jersey_number ? `#${p.jersey_number} ` : ''}{p.name}
                    </option>
                  ))}
                </select>
                <select
                  className="input w-auto text-sm py-1.5"
                  value={selectedEvent}
                  onChange={(e) => setSelectedEvent(e.target.value)}
                >
                  {EVENT_TYPES.map((et) => (
                    <option key={et.value} value={et.value}>{et.emoji} {et.label}</option>
                  ))}
                </select>
                <button type="submit" className="btn-primary btn-sm flex items-center gap-1" disabled={!selectedPlayer}>
                  <Plus size={14} />
                </button>
              </form>
            </div>
          )}
        </div>

        {/* ── Side Panel ────────────────────────────────────────── */}
        <div className="space-y-4">

          {/* Workflow Stepper Card */}
          {match.status === 'analyzed' && (
            <div className="card">
              <h3 className="font-semibold text-sm mb-3 flex items-center gap-2">
                <Rocket size={15} className="text-blue-400" /> Workflow
              </h3>

              {/* Step 1: Identify players */}
              <div className="space-y-3">
                <div className={`flex items-start gap-3 p-3 rounded-xl transition-colors ${
                  assignedCount === 0 ? 'bg-yellow-500/10 border border-yellow-500/20' : 'bg-slate-800/40'
                }`}>
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${
                    assignedCount > 0 ? 'bg-emerald-500 text-white' : 'bg-yellow-500/30 text-yellow-300'
                  }`}>
                    {assignedCount > 0 ? <CheckCircle size={14} /> : '1'}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium">Identifier les joueurs</p>
                    <p className="text-[10px] text-gray-500">
                      {assignedCount === 0
                        ? "Activez le mode Assigner et cliquez sur les joueurs"
                        : `${assignedCount} joueur${assignedCount !== 1 ? 's' : ''} identifié${assignedCount !== 1 ? 's' : ''} sur ${tracks.length} tracks`
                      }
                    </p>
                    {assignedCount === 0 && !assignMode && (
                      <button
                        onClick={() => setAssignMode(true)}
                        className="mt-2 flex items-center gap-1.5 px-3 py-1.5 bg-yellow-500/20 hover:bg-yellow-500/30 border border-yellow-500/30 rounded-lg text-yellow-300 text-xs font-medium transition-colors"
                      >
                        <MousePointer size={12} /> Commencer
                      </button>
                    )}
                  </div>
                </div>

                {/* Step 2: Finalize */}
                <div className={`flex items-start gap-3 p-3 rounded-xl transition-colors ${
                  assignedCount > 0 && !pipelineRunning && !pipelineResult ? 'bg-blue-500/10 border border-blue-500/20' : 'bg-slate-800/40'
                }`}>
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${
                    pipelineResult && !pipelineResult.error ? 'bg-emerald-500 text-white' : 'bg-blue-500/30 text-blue-300'
                  }`}>
                    {pipelineResult && !pipelineResult.error ? <CheckCircle size={14} /> : '2'}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium">Finaliser l'analyse</p>
                    <p className="text-[10px] text-gray-500">
                      Auto-assignation + correction d'identités + détection d'événements
                    </p>

                    {/* Pipeline button */}
                    {assignedCount > 0 && !pipelineRunning && (
                      <button
                        onClick={handleRunPipeline}
                        className="mt-2 w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 rounded-xl text-white text-sm font-medium transition-all shadow-lg shadow-blue-500/20"
                      >
                        <Zap size={15} /> Finaliser l'analyse
                      </button>
                    )}

                    {/* Pipeline progress */}
                    {pipelineRunning && (
                      <div className="mt-2 space-y-2 p-3 bg-slate-800/60 rounded-lg">
                        <div className="flex items-center gap-2">
                          <Loader size={14} className="animate-spin text-blue-400 shrink-0" />
                          <span className="text-xs text-blue-300 flex-1 truncate">
                            {pipelineProgress?.phase || 'Démarrage...'}
                          </span>
                          <span className="text-xs font-mono text-blue-400">
                            {pipelineProgress?.percent || 0}%
                          </span>
                        </div>
                        <div className="w-full bg-slate-700 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-emerald-400 h-full rounded-full transition-all duration-500"
                            style={{ width: `${pipelineProgress?.percent || 0}%` }}
                          />
                        </div>
                        {/* Step indicators */}
                        <div className="flex gap-1.5 text-[10px]">
                          {['Auto-assign', 'Correction', 'Événements'].map((label, i) => (
                            <span key={i} className={`px-1.5 py-0.5 rounded ${
                              (pipelineProgress?.step || 0) > i + 1 ? 'bg-emerald-500/20 text-emerald-400' :
                              (pipelineProgress?.step || 0) === i + 1 ? 'bg-blue-500/20 text-blue-300' :
                              'bg-slate-700/50 text-gray-600'
                            }`}>
                              {label}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Pipeline result */}
                    {pipelineResult && (
                      <div className={`mt-2 text-xs px-3 py-2 rounded-lg ${
                        pipelineResult.error ? 'bg-red-500/10 text-red-400' : 'bg-emerald-500/10 text-emerald-400'
                      }`}>
                        {pipelineResult.error
                          ? pipelineResult.error
                          : `${pipelineResult.events} événements détectés`
                        }
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Reset + export */}
              {assignedCount > 0 && (
                <div className="flex gap-2 mt-3 pt-3 border-t border-slate-700/30">
                  <button
                    onClick={handleResetAssignments}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                  >
                    <Trash2 size={12} /> Reset assignations
                  </button>
                  <button
                    onClick={async () => {
                      const data = await exportMatchData(id)
                      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
                      const url = URL.createObjectURL(blob)
                      const a = document.createElement('a')
                      a.href = url
                      a.download = `match_${id}_export.json`
                      a.click()
                      URL.revokeObjectURL(url)
                    }}
                    className="ml-auto flex items-center gap-1.5 px-3 py-1.5 text-xs text-gray-500 hover:text-gray-300 hover:bg-slate-800 rounded-lg transition-colors"
                  >
                    <Download size={12} /> Export JSON
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Players list card */}
          {match.status === 'analyzed' && assignedCount > 0 && (
            <div className="card">
              <h3 className="font-semibold text-sm mb-3 flex items-center gap-2">
                <UserCheck size={15} className="text-emerald-400" /> Joueurs identifiés
              </h3>
              <div className="space-y-1.5 max-h-[300px] overflow-y-auto pr-1">
                {assignedTracks.map((t) => {
                  const player = players.find(p => p.id === t.player_id)
                  if (!player) return null
                  return (
                    <div
                      key={t.track_id}
                      className="flex items-center gap-2.5 px-3 py-2 bg-slate-800/60 rounded-xl text-sm"
                    >
                      <div
                        className="w-7 h-7 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0"
                        style={{ backgroundColor: player.team_color + '25', color: player.team_color }}
                      >
                        {player.jersey_number ?? '?'}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-sm truncate">{player.name}</p>
                        <p className="text-[10px] text-gray-600">
                          {formatTime(t.first_seen)} - {formatTime(t.last_seen)}
                          {trackConfidence[String(t.track_id)] !== undefined && (
                            <span className={`ml-1 font-mono ${
                              trackConfidence[String(t.track_id)] > 0.7 ? 'text-emerald-500' :
                              trackConfidence[String(t.track_id)] > 0.4 ? 'text-yellow-500' : 'text-red-500'
                            }`}>
                              {Math.round(trackConfidence[String(t.track_id)] * 100)}%
                            </span>
                          )}
                        </p>
                      </div>
                      <button
                        onClick={() => handleUnassignFromOverlay(t.track_id)}
                        className="p-1 text-gray-600 hover:text-red-400 transition-colors"
                      >
                        <X size={12} />
                      </button>
                    </div>
                  )
                })}
              </div>

              {/* Unassigned players hint */}
              {players.filter(p => !assignedPlayerIds.has(p.id)).length > 0 && (
                <div className="mt-3 pt-2 border-t border-slate-700/30">
                  <p className="text-[10px] text-gray-600 mb-1.5">Non identifiés :</p>
                  <div className="flex flex-wrap gap-1">
                    {players.filter(p => !assignedPlayerIds.has(p.id)).map((p) => (
                      <span
                        key={p.id}
                        className="px-1.5 py-0.5 bg-slate-800/40 rounded text-[10px] text-gray-500 flex items-center gap-1"
                      >
                        <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: p.team_color }} />
                        {p.jersey_number ? `#${p.jersey_number} ` : ''}{p.name}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Referee tracks */}
              {refereeTracks.length > 0 && (
                <div className="mt-3 pt-2 border-t border-slate-700/30">
                  {refereeTracks.map((t) => (
                    <div key={t.track_id} className="flex items-center gap-2 px-2 py-1.5 text-xs text-yellow-300">
                      <span>Arbitre</span>
                      <span className="text-gray-600 text-[10px]">#{t.track_id}</span>
                      <button
                        onClick={() => handleUnassignFromOverlay(t.track_id)}
                        className="ml-auto p-0.5 text-gray-600 hover:text-red-400"
                      >
                        <X size={10} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Events Card */}
          <div className="card">
            <h3 className="font-semibold text-sm mb-3 flex items-center gap-2">
              <Target size={15} className="text-red-400" /> Événements
              <span className="text-gray-600 text-xs font-normal">({events.length})</span>
            </h3>
            <div className="space-y-1 max-h-[400px] overflow-y-auto pr-1">
              {events.length === 0 ? (
                <p className="text-gray-600 text-xs text-center py-6">
                  Aucun événement
                </p>
              ) : events.map((ev) => (
                <div
                  key={ev.id}
                  className="flex items-center gap-2 px-2.5 py-1.5 bg-slate-800/60 hover:bg-slate-800 rounded-lg text-sm cursor-pointer transition-colors"
                  onClick={() => seekTo(ev.event_type === 'goal' ? Math.max(0, ev.timestamp_seconds - 5) : ev.timestamp_seconds, true)}
                >
                  <span className="font-mono text-[10px] text-gray-600 w-9 shrink-0">{formatTime(ev.timestamp_seconds)}</span>
                  <span className="shrink-0 text-xs">{EVENT_TYPES.find((e) => e.value === ev.event_type)?.emoji}</span>
                  <div className="flex-1 min-w-0">
                    <span className="font-medium text-sm">{ev.player_name}</span>
                    <span className="text-gray-500 ml-1 text-[10px]">
                      {EVENT_TYPES.find((e) => e.value === ev.event_type)?.label}
                    </span>
                  </div>
                  <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: ev.team_color }} />
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDeleteEvent(ev.id) }}
                    className="text-gray-600 hover:text-red-400 shrink-0"
                  >
                    <Trash2 size={11} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
