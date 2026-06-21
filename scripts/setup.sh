#!/bin/bash
echo "Setting up AstroNova Space Weather Platform..."
python scripts/generate_synthetic_data.py
echo "Setup complete. Run 'docker-compose up --build' to start."
