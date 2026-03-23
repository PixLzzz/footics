# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Footics is a web application for analyzing five-a-side football (futsal) matches. Users upload match videos, the system detects and tracks players using YOLO, and users can log match events (goals, assists, tackles, etc.) to build per-player statistics.

## Architecture

**Monolith deployed as a single Docker image.** The frontend is built at Docker build time and served as static files by the backend.

- **Backend**: FastAPI (Python 3.11) — `backend/`
  - `main.py` — API routes and static file serving
  - `models.py` — SQLAlchemy ORM models (Team, Player, Match, MatchEvent, TrackingFrame, TrackAssignment)
  - `database.py` — SQLite engine/session setup
  - `video_processor.py` — YOLO v8 person detection + ByteTrack multi-object tracking via `ultralytics`
- **Frontend**: React 18 + Vite + Tailwind CSS — `frontend/`
  - `src/api.js` — all API calls centralized here
  - `src/pages/` — Dashboard, MatchUpload, MatchDetail (video player + event logger + tracking panel), TeamsPlayers, PlayerStats, Leaderboard
- **Database**: SQLite stored in `data/footics.db`
- **Video files**: stored in `data/uploads/`

### Key Data Flow

1. User uploads a video → saved to disk, metadata extracted with OpenCV
2. User clicks "Lancer l'analyse YOLO" → background thread runs YOLOv8n tracking, stores bounding boxes in `tracking_frames` table (normalized coords, sampled every 5 frames)
3. User assigns track IDs to known players via the tracking panel
4. User logs events (goals, assists...) at specific timestamps while reviewing the video
5. Stats are aggregated per player from the `match_events` table

## Commands

### Development (local)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev          # Vite dev server on :5173, proxies /api to :8000
```

### Docker

```bash
docker compose up --build        # Build & run on port 8000
docker build -t footics .       # Build image only
docker push <user>/footics      # Push to DockerHub
```

### Build frontend for production

```bash
cd frontend && npm run build     # Outputs to backend/static/
```

## API Structure

All endpoints under `/api/`. Form-encoded POST bodies (not JSON). Key routes:
- `POST /api/matches/upload` — multipart upload (field: `video`)
- `POST /api/matches/{id}/analyze` — triggers async YOLO processing
- `GET /api/matches/{id}/tracks` — list detected track IDs with frame counts
- `POST /api/matches/{id}/assign-track` — map track_id → player_id
- `GET /api/matches/{id}/frame?t=<seconds>` — extract JPEG frame at timestamp
- `GET /api/matches/{id}/video` — stream video file
- `GET /api/players/{id}/stats` — aggregated event counts
- `GET /api/stats/leaderboard?event_type=goal` — top players ranking

## Important Constraints

- Video analysis runs in a **background thread** (not Celery/task queue) — only one analysis at a time is safe
- Tracking data is normalized (0-1 coordinates) relative to video dimensions
- The YOLO model (`yolov8n.pt`) is downloaded at Docker build time
- Frontend build output goes to `backend/static/` which is `.gitignore`d
- SQLite DB path is `./data/footics.db` (relative to backend working dir)
