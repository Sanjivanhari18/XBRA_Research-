#!/bin/bash
export PYTHONPATH="/Users/sanjivan/Documents/XBRA imple"
cd "/Users/sanjivan/Documents/XBRA imple"
exec "/Users/sanjivan/Documents/Gmu Research/triage-poc/.venv/bin/streamlit" run src/xbra/ui/app.py --server.port 8501 --server.headless true
