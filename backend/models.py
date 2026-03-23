from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Enum as SQLEnum
from sqlalchemy.orm import relationship
from database import Base
import enum
from datetime import datetime


class EventType(str, enum.Enum):
    GOAL = "goal"
    ASSIST = "assist"
    SHOT = "shot"
    SHOT_ON_TARGET = "shot_on_target"
    PASS = "pass"
    KEY_PASS = "key_pass"
    INTERCEPTION = "interception"
    TACKLE = "tackle"
    FOUL = "foul"
    DRIBBLE = "dribble"


class MatchStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    ERROR = "error"


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    color = Column(String, default="#3B82F6")
    created_at = Column(DateTime, default=datetime.utcnow)

    players = relationship("Player", back_populates="team")


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    jersey_number = Column(Integer, nullable=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    position = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    team = relationship("Team", back_populates="players")
    events = relationship("MatchEvent", back_populates="player")
    track_assignments = relationship("TrackAssignment", back_populates="player")


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    date = Column(DateTime, default=datetime.utcnow)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    video_path = Column(String, nullable=False)
    video_filename = Column(String, nullable=False)
    duration_seconds = Column(Float, default=0)
    fps = Column(Float, default=0)
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    status = Column(String, default=MatchStatus.UPLOADED)
    created_at = Column(DateTime, default=datetime.utcnow)

    team = relationship("Team", foreign_keys=[team_id])
    events = relationship("MatchEvent", back_populates="match", cascade="all, delete-orphan")
    tracking_data = relationship("TrackingFrame", back_populates="match", cascade="all, delete-orphan")
    ball_data = relationship("BallFrame", back_populates="match", cascade="all, delete-orphan")
    track_assignments = relationship("TrackAssignment", back_populates="match", cascade="all, delete-orphan")


class MatchEvent(Base):
    __tablename__ = "match_events"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    event_type = Column(String, nullable=False)
    timestamp_seconds = Column(Float, nullable=False)
    x = Column(Float, nullable=True)
    y = Column(Float, nullable=True)
    notes = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="events")
    player = relationship("Player", back_populates="events")


class TrackingFrame(Base):
    __tablename__ = "tracking_frames"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    frame_number = Column(Integer, nullable=False)
    timestamp_seconds = Column(Float, nullable=False)
    track_id = Column(Integer, nullable=False)
    bbox_x = Column(Float, nullable=False)
    bbox_y = Column(Float, nullable=False)
    bbox_w = Column(Float, nullable=False)
    bbox_h = Column(Float, nullable=False)
    confidence = Column(Float, default=0)

    match = relationship("Match", back_populates="tracking_data")


class BallFrame(Base):
    __tablename__ = "ball_frames"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    frame_number = Column(Integer, nullable=False)
    timestamp_seconds = Column(Float, nullable=False)
    x = Column(Float, nullable=False)  # normalized center x
    y = Column(Float, nullable=False)  # normalized center y
    confidence = Column(Float, default=0)

    match = relationship("Match", back_populates="ball_data")


class TrackAssignment(Base):
    __tablename__ = "track_assignments"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    track_id = Column(Integer, nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=True)
    is_referee = Column(Boolean, default=False)

    match = relationship("Match", back_populates="track_assignments")
    player = relationship("Player", back_populates="track_assignments")
