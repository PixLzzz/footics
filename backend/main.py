import os
import shutil
import uuid
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from database import engine, get_db, Base
from models import (
    Team, Player, Match, MatchEvent, TrackingFrame, BallFrame, TrackAssignment,
    MatchStatus, EventType,
)

try:
    import cv2
    from video_processor import process_video, get_video_info, extract_frame
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Create tables
Base.metadata.create_all(bind=engine)

# ── DB migration: team_home_id/team_away_id → team_id ─────────────────────
def _migrate_matches_single_team():
    """Migrate old matches table (home/away) to single team_id."""
    from sqlalchemy import text, inspect
    with engine.connect() as conn:
        inspector = inspect(engine)
        columns = [c["name"] for c in inspector.get_columns("matches")]
        if "team_home_id" in columns and "team_id" not in columns:
            conn.execute(text("ALTER TABLE matches ADD COLUMN team_id INTEGER REFERENCES teams(id)"))
            conn.execute(text("UPDATE matches SET team_id = team_home_id"))
            conn.commit()

try:
    _migrate_matches_single_team()
except Exception:
    pass  # fresh DB or already migrated

# Ensure directories exist
UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Footics - Five-a-side Match Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Teams ───────────────────────────────────────────────────────────────────

@app.post("/api/teams")
def create_team(name: str = Form(...), color: str = Form("#3B82F6"), db: Session = Depends(get_db)):
    team = Team(name=name, color=color)
    db.add(team)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, f"Une équipe nommée '{name}' existe déjà")
    db.refresh(team)
    return {"id": team.id, "name": team.name, "color": team.color}


@app.get("/api/teams")
def list_teams(db: Session = Depends(get_db)):
    teams = db.query(Team).all()
    return [
        {
            "id": t.id, "name": t.name, "color": t.color,
            "player_count": len(t.players),
        }
        for t in teams
    ]


@app.put("/api/teams/{team_id}")
def update_team(
    team_id: int,
    name: str = Form(...),
    color: str = Form("#3B82F6"),
    db: Session = Depends(get_db),
):
    team = db.query(Team).filter(Team.id == team_id).first()
    if not team:
        raise HTTPException(404, "Team not found")
    team.name = name
    team.color = color
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, f"Une équipe nommée '{name}' existe déjà")
    db.refresh(team)
    return {"id": team.id, "name": team.name, "color": team.color}


@app.delete("/api/teams/{team_id}")
def delete_team(team_id: int, db: Session = Depends(get_db)):
    team = db.query(Team).filter(Team.id == team_id).first()
    if not team:
        raise HTTPException(404, "Team not found")
    db.delete(team)
    db.commit()
    return {"ok": True}

# ─── Players ─────────────────────────────────────────────────────────────────

@app.post("/api/players")
def create_player(
    name: str = Form(...),
    team_id: int = Form(...),
    jersey_number: Optional[int] = Form(None),
    position: str = Form(""),
    db: Session = Depends(get_db),
):
    player = Player(name=name, team_id=team_id, jersey_number=jersey_number, position=position)
    db.add(player)
    db.commit()
    db.refresh(player)
    return {
        "id": player.id, "name": player.name,
        "jersey_number": player.jersey_number,
        "team_id": player.team_id, "position": player.position,
    }


@app.put("/api/players/{player_id}")
def update_player(
    player_id: int,
    name: str = Form(...),
    team_id: int = Form(...),
    jersey_number: Optional[int] = Form(None),
    position: str = Form(""),
    db: Session = Depends(get_db),
):
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        raise HTTPException(404, "Player not found")
    player.name = name
    player.team_id = team_id
    player.jersey_number = jersey_number
    player.position = position
    db.commit()
    db.refresh(player)
    return {
        "id": player.id, "name": player.name,
        "jersey_number": player.jersey_number,
        "team_id": player.team_id, "position": player.position,
    }


@app.get("/api/players")
def list_players(team_id: Optional[int] = Query(None), db: Session = Depends(get_db)):
    q = db.query(Player)
    if team_id:
        q = q.filter(Player.team_id == team_id)
    players = q.all()
    return [
        {
            "id": p.id, "name": p.name, "jersey_number": p.jersey_number,
            "team_id": p.team_id, "team_name": p.team.name,
            "team_color": p.team.color, "position": p.position,
        }
        for p in players
    ]


@app.delete("/api/players/{player_id}")
def delete_player(player_id: int, db: Session = Depends(get_db)):
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        raise HTTPException(404, "Player not found")
    db.delete(player)
    db.commit()
    return {"ok": True}

# ─── Matches ─────────────────────────────────────────────────────────────────

@app.post("/api/matches/upload")
def upload_match(
    title: str = Form(...),
    video: UploadFile = File(...),
    team_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
):
    ext = Path(video.filename).suffix.lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        raise HTTPException(400, "Format vidéo non supporté (MP4, AVI, MOV, MKV, WebM)")

    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = UPLOAD_DIR / filename

    with open(filepath, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Get video info
    info = {"fps": 0, "duration_seconds": 0, "width": 0, "height": 0}
    if HAS_CV2:
        try:
            info = get_video_info(str(filepath))
        except Exception:
            pass

    match = Match(
        title=title,
        video_path=str(filepath),
        video_filename=video.filename,
        team_id=team_id if team_id else None,
        fps=info["fps"],
        duration_seconds=info["duration_seconds"],
        width=info["width"],
        height=info["height"],
    )
    db.add(match)
    db.commit()
    db.refresh(match)
    return {"id": match.id, "title": match.title, "status": match.status}


@app.get("/api/matches")
def list_matches(db: Session = Depends(get_db)):
    matches = db.query(Match).order_by(Match.created_at.desc()).all()
    return [
        {
            "id": m.id, "title": m.title, "date": m.date.isoformat(),
            "status": m.status, "duration_seconds": m.duration_seconds,
            "video_filename": m.video_filename,
            "team": {"id": m.team.id, "name": m.team.name, "color": m.team.color} if m.team else None,
            "event_count": len(m.events),
        }
        for m in matches
    ]


@app.get("/api/matches/{match_id}")
def get_match(match_id: int, db: Session = Depends(get_db)):
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Match not found")
    return {
        "id": match.id, "title": match.title, "date": match.date.isoformat(),
        "status": match.status, "duration_seconds": match.duration_seconds,
        "fps": match.fps, "width": match.width, "height": match.height,
        "video_filename": match.video_filename,
        "team": {"id": match.team.id, "name": match.team.name, "color": match.team.color} if match.team else None,
    }


@app.delete("/api/matches/{match_id}")
def delete_match(match_id: int, db: Session = Depends(get_db)):
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Match not found")
    # Delete video file
    try:
        os.unlink(match.video_path)
    except OSError:
        pass
    db.delete(match)
    db.commit()
    return {"ok": True}


@app.get("/api/matches/{match_id}/video")
def stream_video(match_id: int, db: Session = Depends(get_db)):
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Match not found")
    if not os.path.exists(match.video_path):
        raise HTTPException(404, "Video file not found")
    return FileResponse(match.video_path, media_type="video/mp4")


@app.get("/api/matches/{match_id}/frame")
def get_frame(match_id: int, t: float = Query(0), db: Session = Depends(get_db)):
    """Get a JPEG frame at timestamp t (seconds)."""
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Match not found")
    if not HAS_CV2:
        raise HTTPException(501, "OpenCV not installed")
    frame = extract_frame(match.video_path, t)
    if frame is None:
        raise HTTPException(400, "Could not extract frame")
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(content=buf.tobytes(), media_type="image/jpeg")

# ─── Video Analysis ──────────────────────────────────────────────────────────

@app.post("/api/matches/{match_id}/analyze")
def analyze_match(match_id: int, db: Session = Depends(get_db)):
    """Trigger YOLO player tracking analysis (runs in background thread)."""
    if not HAS_CV2:
        raise HTTPException(501, "OpenCV/YOLO not installed")
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Match not found")
    if match.status == MatchStatus.PROCESSING:
        raise HTTPException(400, "Analysis already in progress")

    # Reset status so analysis can restart (handles error/analyzed states)
    match.status = MatchStatus.PROCESSING
    db.commit()

    from database import SessionLocal

    def run_analysis():
        session = SessionLocal()
        try:
            process_video(match_id, session)
        finally:
            session.close()

    thread = threading.Thread(target=run_analysis, daemon=True)
    thread.start()

    return {"status": "processing", "message": "Analysis started"}


@app.post("/api/matches/{match_id}/cancel-analysis")
def cancel_analysis(match_id: int, db: Session = Depends(get_db)):
    """Cancel a running YOLO analysis. Also resets status if thread is dead."""
    from video_processor import analysis_cancel, analysis_progress
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Match not found")
    if match.status != MatchStatus.PROCESSING:
        raise HTTPException(400, "No analysis in progress")
    # Signal the thread to stop
    analysis_cancel.add(match_id)
    # Also force-reset status in case the thread already crashed
    match.status = MatchStatus.UPLOADED
    db.commit()
    analysis_progress.pop(match_id, None)
    return {"ok": True, "message": "Analysis cancelled"}


@app.get("/api/matches/{match_id}/analysis-progress")
def get_analysis_progress(match_id: int, db: Session = Depends(get_db)):
    """Get the current progress of YOLO analysis."""
    from video_processor import analysis_progress
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Match not found")
    progress = analysis_progress.get(match_id, {"current": 0, "total": 0, "percent": 0})
    return {"status": match.status, **progress}


@app.post("/api/matches/{match_id}/detect-events")
def auto_detect_events(
    match_id: int,
    db: Session = Depends(get_db),
):
    """Auto-detect match events from tracking + ball data (camera-angle agnostic)."""
    from event_detector import detect_events
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Match not found")
    if match.status != "analyzed":
        raise HTTPException(400, "Match must be analyzed first")

    # Check ball data exists
    ball_count = db.query(BallFrame).filter_by(match_id=match_id).count()
    if ball_count == 0:
        raise HTTPException(400, "Aucune donnée de ballon détectée. Relancez l'analyse.")

    # Check assignments exist
    assign_count = db.query(TrackAssignment).filter_by(match_id=match_id).count()
    if assign_count == 0:
        raise HTTPException(400, "Assignez d'abord les joueurs aux tracks détectés.")

    events = detect_events(match_id, db)
    return {
        "count": len(events),
        "events": [
            {"event_type": e["event_type"], "player_id": e["player_id"],
             "timestamp_seconds": e["timestamp_seconds"]}
            for e in events
        ],
    }


@app.get("/api/matches/{match_id}/team-labels")
def get_team_labels_endpoint(match_id: int, db: Session = Depends(get_db)):
    """Get team classification labels for all tracks."""
    from team_classifier import get_team_labels
    labels = get_team_labels(match_id)
    if not labels:
        return {"labels": {}, "message": "No team classification available"}
    return {"labels": {str(k): v for k, v in labels.items()}}


@app.get("/api/matches/{match_id}/ball-stats")
def get_ball_stats(match_id: int, db: Session = Depends(get_db)):
    """Get ball detection stats for a match."""
    count = db.query(BallFrame).filter_by(match_id=match_id).count()
    return {"match_id": match_id, "ball_detections": count}


@app.get("/api/matches/{match_id}/tracking")
def get_tracking_data(
    match_id: int,
    start: float = Query(0),
    end: float = Query(None),
    db: Session = Depends(get_db),
):
    """Get tracking data for a time range."""
    q = db.query(TrackingFrame).filter(
        TrackingFrame.match_id == match_id,
        TrackingFrame.timestamp_seconds >= start,
    )
    if end is not None:
        q = q.filter(TrackingFrame.timestamp_seconds <= end)
    q = q.order_by(TrackingFrame.timestamp_seconds, TrackingFrame.track_id)

    frames = q.limit(5000).all()

    # Group by timestamp
    grouped = {}
    for f in frames:
        ts = f.timestamp_seconds
        if ts not in grouped:
            grouped[ts] = []
        grouped[ts].append({
            "track_id": f.track_id,
            "bbox": [f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h],
            "confidence": f.confidence,
        })

    return {"frames": grouped}


@app.get("/api/matches/{match_id}/tracking-bulk")
def get_tracking_bulk(
    match_id: int,
    start: float = Query(0),
    end: float = Query(10),
    db: Session = Depends(get_db),
):
    """Get all tracking frames + assignments for a time window.

    Returns sorted timestamps, boxes grouped by timestamp, and assignment map.
    Designed for client-side pre-fetching in ~10s chunks.
    """
    frames = (
        db.query(TrackingFrame)
        .filter(
            TrackingFrame.match_id == match_id,
            TrackingFrame.timestamp_seconds >= start,
            TrackingFrame.timestamp_seconds <= end,
        )
        .order_by(TrackingFrame.timestamp_seconds, TrackingFrame.track_id)
        .all()
    )

    # Group by timestamp
    grouped = {}
    for f in frames:
        ts = f.timestamp_seconds
        if ts not in grouped:
            grouped[ts] = []
        grouped[ts].append({
            "track_id": f.track_id,
            "bbox": [f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h],
        })

    # Sorted list of timestamps for binary search on client
    timestamps = sorted(grouped.keys())

    # Load assignments once
    assignments = {}
    referee_tracks = []
    for a in db.query(TrackAssignment).filter_by(match_id=match_id).all():
        if a.is_referee:
            referee_tracks.append(a.track_id)
        elif a.player_id:
            player = db.query(Player).filter_by(id=a.player_id).first()
            if player:
                assignments[str(a.track_id)] = {
                    "player_id": player.id,
                    "name": player.name,
                    "jersey": player.jersey_number,
                    "team_color": player.team.color,
                    "team_name": player.team.name,
                }

    # Load team labels for overlay coloring
    try:
        from team_classifier import get_team_labels
        team_labels = get_team_labels(match_id)
        team_labels_str = {str(k): v for k, v in team_labels.items()}
    except Exception:
        team_labels_str = {}

    return {
        "timestamps": timestamps,
        "frames": grouped,
        "assignments": assignments,
        "referee_tracks": referee_tracks,
        "team_labels": team_labels_str,
        "start": start,
        "end": end,
    }


@app.get("/api/matches/{match_id}/tracking-at")
def get_tracking_at_time(
    match_id: int,
    t: float = Query(...),
    db: Session = Depends(get_db),
):
    """Get bounding boxes at the two nearest tracked timestamps for interpolation.

    Returns `before` and `after` frame data plus a `ratio` (0-1) for lerp.
    If exact match, ratio=0 and after is empty.
    """
    # Find the frame just before and just after t
    ts_before = (
        db.query(TrackingFrame.timestamp_seconds)
        .filter(TrackingFrame.match_id == match_id, TrackingFrame.timestamp_seconds <= t)
        .order_by(TrackingFrame.timestamp_seconds.desc())
        .first()
    )
    ts_after = (
        db.query(TrackingFrame.timestamp_seconds)
        .filter(TrackingFrame.match_id == match_id, TrackingFrame.timestamp_seconds > t)
        .order_by(TrackingFrame.timestamp_seconds.asc())
        .first()
    )

    if not ts_before and not ts_after:
        return {"before": {"timestamp": t, "boxes": []}, "after": None, "ratio": 0}

    # Load assignments (shared for both frames)
    assignments = {}
    referee_tracks = set()
    for a in db.query(TrackAssignment).filter_by(match_id=match_id).all():
        if a.is_referee:
            referee_tracks.add(a.track_id)
        elif a.player_id:
            player = db.query(Player).filter_by(id=a.player_id).first()
            if player:
                assignments[a.track_id] = {
                    "player_id": player.id,
                    "name": player.name,
                    "jersey": player.jersey_number,
                    "team_color": player.team.color,
                    "team_name": player.team.name,
                }

    def build_boxes(ts_val):
        if ts_val is None:
            return None
        ts = ts_val[0]
        frames = db.query(TrackingFrame).filter_by(match_id=match_id, timestamp_seconds=ts).all()
        boxes = []
        for f in frames:
            box = {"track_id": f.track_id, "bbox": [f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h]}
            if f.track_id in referee_tracks:
                box["is_referee"] = True
            elif f.track_id in assignments:
                box["player"] = assignments[f.track_id]
            boxes.append(box)
        return {"timestamp": ts, "boxes": boxes}

    before = build_boxes(ts_before) or build_boxes(ts_after)
    after = build_boxes(ts_after) if ts_before else None

    # Compute interpolation ratio
    ratio = 0
    if before and after and after["timestamp"] != before["timestamp"]:
        ratio = (t - before["timestamp"]) / (after["timestamp"] - before["timestamp"])
        ratio = max(0, min(1, ratio))

    return {"before": before, "after": after, "ratio": ratio}


@app.delete("/api/matches/{match_id}/unassign-track")
def unassign_track(
    match_id: int,
    track_id: int = Query(...),
    db: Session = Depends(get_db),
):
    """Remove a track-player assignment."""
    deleted = (
        db.query(TrackAssignment)
        .filter(TrackAssignment.match_id == match_id, TrackAssignment.track_id == track_id)
        .delete()
    )
    db.commit()
    return {"ok": True, "deleted": deleted}


@app.delete("/api/matches/{match_id}/unassign-all")
def unassign_all_tracks(
    match_id: int,
    db: Session = Depends(get_db),
):
    """Remove ALL track-player assignments for a match."""
    deleted = (
        db.query(TrackAssignment)
        .filter(TrackAssignment.match_id == match_id)
        .delete()
    )
    db.commit()
    return {"ok": True, "deleted": deleted}


@app.get("/api/matches/{match_id}/tracks")
def get_unique_tracks(
    match_id: int,
    min_frames: int = Query(0),
    db: Session = Depends(get_db),
):
    """Get list of unique track IDs found in the video."""
    tracks = (
        db.query(
            TrackingFrame.track_id,
            func.count(TrackingFrame.id).label("frame_count"),
            func.min(TrackingFrame.timestamp_seconds).label("first_seen"),
            func.max(TrackingFrame.timestamp_seconds).label("last_seen"),
        )
        .filter(TrackingFrame.match_id == match_id)
        .group_by(TrackingFrame.track_id)
        .having(func.count(TrackingFrame.id) >= min_frames)
        .order_by(func.count(TrackingFrame.id).desc())
        .all()
    )

    # Get existing assignments
    assignments = (
        db.query(TrackAssignment)
        .filter(TrackAssignment.match_id == match_id)
        .all()
    )
    assignment_map = {a.track_id: a.player_id for a in assignments}
    referee_map = {a.track_id: a.is_referee for a in assignments}

    return [
        {
            "track_id": t.track_id,
            "frame_count": t.frame_count,
            "first_seen": t.first_seen,
            "last_seen": t.last_seen,
            "player_id": assignment_map.get(t.track_id),
            "is_referee": referee_map.get(t.track_id, False),
        }
        for t in tracks
    ]


@app.get("/api/matches/{match_id}/track-thumbnail")
def get_track_thumbnail(
    match_id: int,
    track_id: int = Query(...),
    db: Session = Depends(get_db),
):
    """Get a cropped JPEG thumbnail of a tracked person."""
    if not HAS_CV2:
        raise HTTPException(501, "OpenCV not installed")

    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Match not found")

    # Get a frame from the middle of this track's appearance
    track_frame = (
        db.query(TrackingFrame)
        .filter_by(match_id=match_id, track_id=track_id)
        .order_by(TrackingFrame.timestamp_seconds)
        .all()
    )
    if not track_frame:
        raise HTTPException(404, "Track not found")

    # Pick the middle frame for best representation
    mid = track_frame[len(track_frame) // 2]

    frame = extract_frame(match.video_path, mid.timestamp_seconds)
    if frame is None:
        raise HTTPException(400, "Could not extract frame")

    h, w = frame.shape[:2]
    # Denormalize bbox and add padding
    pad = 0.3  # 30% padding around the player
    x1 = int(max(0, (mid.bbox_x - mid.bbox_w * pad) * w))
    y1 = int(max(0, (mid.bbox_y - mid.bbox_h * pad) * h))
    x2 = int(min(w, (mid.bbox_x + mid.bbox_w * (1 + pad)) * w))
    y2 = int(min(h, (mid.bbox_y + mid.bbox_h * (1 + pad)) * h))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise HTTPException(400, "Empty crop")

    # Resize to max 150px height
    target_h = 150
    scale = target_h / crop.shape[0]
    crop = cv2.resize(crop, (int(crop.shape[1] * scale), target_h))

    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buf.tobytes(), media_type="image/jpeg")

# ─── Identity Correction ────────────────────────────────────────────────────

@app.post("/api/matches/{match_id}/correct-identities")
def correct_identities_endpoint(
    match_id: int,
    db: Session = Depends(get_db),
):
    """Re-identify all detections frame-by-frame using player appearance (background thread)."""
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Match not found")
    if match.status != "analyzed":
        raise HTTPException(400, "Match must be analyzed first")

    from identity_corrector import correction_progress
    prog = correction_progress.get(match_id)
    if prog and not prog.get("done", True) and prog.get("percent", 0) > 0:
        raise HTTPException(400, "Correction déjà en cours")

    correction_progress[match_id] = {
        "percent": 0, "phase": "Démarrage...", "done": False,
    }

    from database import SessionLocal

    def run_correction():
        session = SessionLocal()
        try:
            from identity_corrector import correct_identities
            result = correct_identities(match_id, session)
            correction_progress[match_id] = {
                "percent": 100, "phase": "Terminé", "done": True,
                "corrected": result.get("corrected_count", 0),
                "message": result.get("error", ""),
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            correction_progress[match_id] = {
                "percent": 100, "phase": f"Erreur: {e}", "done": True, "error": str(e),
            }
        finally:
            session.close()

    thread = threading.Thread(target=run_correction, daemon=True)
    thread.start()

    return {"status": "processing", "message": "Correction des identités lancée"}


@app.get("/api/matches/{match_id}/correction-progress")
def get_correction_progress(match_id: int):
    from identity_corrector import correction_progress
    return correction_progress.get(match_id, {
        "percent": 0, "phase": "", "done": True,
    })


# ─── Track ↔ Player Assignment ──────────────────────────────────────────────

@app.post("/api/matches/{match_id}/auto-assign-tracks")
def auto_assign_tracks_endpoint(
    match_id: int,
    threshold: float = Form(0.38),
    db: Session = Depends(get_db),
):
    """Auto-assign unassigned tracks using multi-feature appearance matching (background thread)."""
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Match not found")
    if match.status != "analyzed":
        raise HTTPException(400, "Match must be analyzed first")

    from appearance import assignment_progress
    # Check if already running
    prog = assignment_progress.get(match_id)
    if prog and 0 < prog.get("percent", 100) < 100:
        raise HTTPException(400, "Auto-assignation déjà en cours")

    # Set initial progress BEFORE starting the thread
    assignment_progress[match_id] = {
        "current": 0, "total": 0, "percent": 0,
        "phase": "Démarrage...", "done": False,
    }

    from database import SessionLocal

    def run_auto_assign():
        session = SessionLocal()
        try:
            from appearance import auto_assign_tracks
            result = auto_assign_tracks(match_id, session, threshold=threshold)
            # Mark as done
            assignment_progress[match_id] = {
                "current": 1, "total": 1, "percent": 100,
                "phase": "Terminé", "done": True,
                "assigned_count": result.get("assigned_count", 0),
                "message": result.get("message", ""),
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            assignment_progress[match_id] = {
                "current": 0, "total": 0, "percent": 100,
                "phase": f"Erreur: {e}", "done": True, "error": str(e),
            }
        finally:
            session.close()

    thread = threading.Thread(target=run_auto_assign, daemon=True)
    thread.start()

    return {"status": "processing", "message": "Auto-assignation lancée"}


@app.get("/api/matches/{match_id}/auto-assign-progress")
def get_auto_assign_progress(match_id: int):
    """Get the current progress of auto-assignment."""
    from appearance import assignment_progress
    progress = assignment_progress.get(match_id, {
        "current": 0, "total": 0, "percent": 0, "phase": "",
    })
    return progress


@app.get("/api/matches/{match_id}/auto-assign-result")
def get_auto_assign_result(match_id: int, db: Session = Depends(get_db)):
    """Check if auto-assignment is done and return the result summary."""
    from appearance import assignment_progress
    progress = assignment_progress.get(match_id)
    done = progress is not None and progress.get("percent", 0) >= 100
    # Count assignments
    count = db.query(TrackAssignment).filter_by(match_id=match_id).count()
    return {"done": done, "total_assignments": count, "progress": progress}


@app.post("/api/matches/{match_id}/assign-track")
def assign_track(
    match_id: int,
    track_id: int = Form(...),
    player_id: Optional[int] = Form(None),
    is_referee: bool = Form(False),
    db: Session = Depends(get_db),
):
    """Assign a tracking ID to a player, or mark as referee."""
    existing = (
        db.query(TrackAssignment)
        .filter(TrackAssignment.match_id == match_id, TrackAssignment.track_id == track_id)
        .first()
    )
    if existing:
        existing.player_id = player_id
        existing.is_referee = is_referee
    else:
        db.add(TrackAssignment(
            match_id=match_id, track_id=track_id,
            player_id=player_id, is_referee=is_referee,
        ))
    db.commit()
    return {"ok": True}

# ─── Match Events ────────────────────────────────────────────────────────────

@app.post("/api/matches/{match_id}/events")
def create_event(
    match_id: int,
    player_id: int = Form(...),
    event_type: str = Form(...),
    timestamp_seconds: float = Form(...),
    x: Optional[float] = Form(None),
    y: Optional[float] = Form(None),
    notes: str = Form(""),
    db: Session = Depends(get_db),
):
    event = MatchEvent(
        match_id=match_id, player_id=player_id,
        event_type=event_type, timestamp_seconds=timestamp_seconds,
        x=x, y=y, notes=notes,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return {
        "id": event.id, "event_type": event.event_type,
        "player_id": event.player_id, "timestamp_seconds": event.timestamp_seconds,
    }


@app.get("/api/matches/{match_id}/events")
def list_events(match_id: int, db: Session = Depends(get_db)):
    events = (
        db.query(MatchEvent)
        .filter(MatchEvent.match_id == match_id)
        .order_by(MatchEvent.timestamp_seconds)
        .all()
    )
    return [
        {
            "id": e.id, "event_type": e.event_type,
            "player_id": e.player_id, "player_name": e.player.name,
            "team_name": e.player.team.name, "team_color": e.player.team.color,
            "timestamp_seconds": e.timestamp_seconds,
            "x": e.x, "y": e.y, "notes": e.notes,
        }
        for e in events
    ]


@app.delete("/api/events/{event_id}")
def delete_event(event_id: int, db: Session = Depends(get_db)):
    event = db.query(MatchEvent).filter(MatchEvent.id == event_id).first()
    if not event:
        raise HTTPException(404, "Event not found")
    db.delete(event)
    db.commit()
    return {"ok": True}

# ─── Player Stats ────────────────────────────────────────────────────────────

@app.get("/api/players/{player_id}/stats")
def player_stats(player_id: int, db: Session = Depends(get_db)):
    player = db.query(Player).filter(Player.id == player_id).first()
    if not player:
        raise HTTPException(404, "Player not found")

    stats = (
        db.query(MatchEvent.event_type, func.count(MatchEvent.id))
        .filter(MatchEvent.player_id == player_id)
        .group_by(MatchEvent.event_type)
        .all()
    )

    matches_played = (
        db.query(func.count(func.distinct(MatchEvent.match_id)))
        .filter(MatchEvent.player_id == player_id)
        .scalar()
    )

    return {
        "player": {
            "id": player.id, "name": player.name,
            "jersey_number": player.jersey_number,
            "team_name": player.team.name, "team_color": player.team.color,
        },
        "matches_played": matches_played,
        "stats": {event_type: count for event_type, count in stats},
    }


@app.get("/api/stats/leaderboard")
def leaderboard(event_type: str = Query("goal"), db: Session = Depends(get_db)):
    results = (
        db.query(
            Player.id, Player.name, Player.jersey_number,
            Team.name.label("team_name"), Team.color.label("team_color"),
            func.count(MatchEvent.id).label("count"),
        )
        .join(MatchEvent, MatchEvent.player_id == Player.id)
        .join(Team, Team.id == Player.team_id)
        .filter(MatchEvent.event_type == event_type)
        .group_by(Player.id)
        .order_by(func.count(MatchEvent.id).desc())
        .limit(20)
        .all()
    )
    return [
        {
            "player_id": r.id, "name": r.name,
            "jersey_number": r.jersey_number,
            "team_name": r.team_name, "team_color": r.team_color,
            "count": r.count,
        }
        for r in results
    ]


# ─── Serve Frontend ──────────────────────────────────────────────────────────

frontend_dir = Path("./static")
if frontend_dir.exists():
    from fastapi.responses import HTMLResponse

    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

    # Catch-all: serve index.html for any non-API route (SPA client-side routing)
    @app.get("/{path:path}", response_class=HTMLResponse)
    def serve_spa(path: str):
        index = frontend_dir / "index.html"
        return HTMLResponse(content=index.read_text())
