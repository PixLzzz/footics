# Footics ⚽

Analyse de matchs de five (futsal) avec tracking automatique des joueurs par intelligence artificielle.

## Fonctionnalités

- **Upload vidéo** — Importez vos matchs de five directement depuis le navigateur
- **Tracking IA (YOLOv8)** — Détection et suivi automatique des joueurs image par image
- **Identification sur vidéo** — Cliquez directement sur les joueurs dans la vidéo pour les identifier
- **Détection d'événements** — Goals, tirs, passes, dribbles, interceptions détectés automatiquement
- **Statistiques joueurs** — Stats individuelles agrégées sur tous les matchs
- **Classement** — Leaderboard par type d'événement (buts, passes décisives, etc.)
- **Gestion arbitre** — Excluez l'arbitre du tracking et des statistiques

## Stack technique

| Composant | Technologie |
|-----------|------------|
| Backend | FastAPI (Python 3.11) |
| Frontend | React 18 + Vite + Tailwind CSS |
| Base de données | SQLite |
| Tracking | YOLOv8n + ByteTrack |
| Vision | OpenCV |

## Installation

### Docker (recommandé)

```bash
docker compose up --build
```

L'application est accessible sur `http://localhost:8000`.

### Développement local

```bash
# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (autre terminal)
cd frontend
npm install
npm run dev
```

Le frontend tourne sur `:5173` et proxy les appels `/api` vers `:8000`.

## Architecture

```
footics/
├── backend/
│   ├── main.py              # Routes API FastAPI
│   ├── models.py            # Modèles SQLAlchemy
│   ├── database.py          # Configuration SQLite
│   ├── video_processor.py   # Pipeline YOLO + ByteTrack
│   └── event_detector.py    # Détection automatique d'événements
├── frontend/
│   └── src/
│       ├── api.js           # Client API centralisé
│       └── pages/           # Dashboard, MatchDetail, TeamsPlayers, etc.
├── Dockerfile               # Build multi-stage (Node + Python)
└── docker-compose.yml
```

## Utilisation

1. **Créez des équipes** et ajoutez des joueurs (nom, numéro, position)
2. **Uploadez une vidéo** de match
3. **Lancez l'analyse YOLO** pour détecter les joueurs
4. **Identifiez les joueurs** en cliquant sur eux dans la vidéo
5. **Détectez les événements** automatiquement ou ajoutez-les manuellement
6. **Consultez les stats** par joueur et le classement global

## License

MIT
