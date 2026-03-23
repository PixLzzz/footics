const BASE = '/api';

async function request(url, options = {}) {
  const res = await fetch(`${BASE}${url}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

function formData(obj) {
  const fd = new FormData();
  for (const [k, v] of Object.entries(obj)) {
    if (v !== null && v !== undefined && v !== '') fd.append(k, v);
  }
  return fd;
}

// Teams
export const getTeams = () => request('/teams');
export const createTeam = (data) => request('/teams', { method: 'POST', body: formData(data) });
export const updateTeam = (id, data) => request(`/teams/${id}`, { method: 'PUT', body: formData(data) });
export const deleteTeam = (id) => request(`/teams/${id}`, { method: 'DELETE' });

// Players
export const getPlayers = (teamId) => request(`/players${teamId ? `?team_id=${teamId}` : ''}`);
export const createPlayer = (data) => request('/players', { method: 'POST', body: formData(data) });
export const updatePlayer = (id, data) => request(`/players/${id}`, { method: 'PUT', body: formData(data) });
export const deletePlayer = (id) => request(`/players/${id}`, { method: 'DELETE' });

// Matches
export const getMatches = () => request('/matches');
export const getMatch = (id) => request(`/matches/${id}`);
export const deleteMatch = (id) => request(`/matches/${id}`, { method: 'DELETE' });
export const uploadMatch = (formDataObj) =>
  request('/matches/upload', { method: 'POST', body: formDataObj });

// Analysis
export const analyzeMatch = (id) => request(`/matches/${id}/analyze`, { method: 'POST' });
export const getAnalysisProgress = (id) => request(`/matches/${id}/analysis-progress`);
export const detectEvents = (matchId, homeAttacksRight) =>
  request(`/matches/${matchId}/detect-events`, {
    method: 'POST',
    body: formData({ home_attacks_right: homeAttacksRight }),
  });
export const getBallStats = (id) => request(`/matches/${id}/ball-stats`);
export const getTracking = (id, start = 0, end = null) =>
  request(`/matches/${id}/tracking?start=${start}${end !== null ? `&end=${end}` : ''}`);
export const getTracks = (id, minFrames = 0) => request(`/matches/${id}/tracks?min_frames=${minFrames}`);
export const trackThumbnailUrl = (matchId, trackId) => `${BASE}/matches/${matchId}/track-thumbnail?track_id=${trackId}`;
export const assignTrack = (matchId, data) =>
  request(`/matches/${matchId}/assign-track`, { method: 'POST', body: formData(data) });
export const markReferee = (matchId, trackId) =>
  request(`/matches/${matchId}/assign-track`, { method: 'POST', body: formData({ track_id: trackId, is_referee: true }) });
export const unassignTrack = (matchId, trackId) =>
  request(`/matches/${matchId}/unassign-track?track_id=${trackId}`, { method: 'DELETE' });
export const getTrackingAt = (matchId, t) => request(`/matches/${matchId}/tracking-at?t=${t}`);

// Events
export const getEvents = (matchId) => request(`/matches/${matchId}/events`);
export const createEvent = (matchId, data) =>
  request(`/matches/${matchId}/events`, { method: 'POST', body: formData(data) });
export const deleteEvent = (id) => request(`/events/${id}`, { method: 'DELETE' });

// Stats
export const getPlayerStats = (id) => request(`/players/${id}/stats`);
export const getLeaderboard = (eventType = 'goal') => request(`/stats/leaderboard?event_type=${eventType}`);

// URLs
export const videoUrl = (matchId) => `${BASE}/matches/${matchId}/video`;
export const frameUrl = (matchId, t) => `${BASE}/matches/${matchId}/frame?t=${t}`;
