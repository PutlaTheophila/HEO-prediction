#!/bin/bash
# CLI Predictor Launcher Script

cd "$(dirname "$0")"
source venv/bin/activate
python predict_cli.py "$@"
