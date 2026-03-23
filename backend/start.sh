#!/bin/bash
cd /home/pixl/Dev/fou2foot/backend
./venv/bin/python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 2>&1 || echo "BACKEND FAILED WITH CODE $?"
