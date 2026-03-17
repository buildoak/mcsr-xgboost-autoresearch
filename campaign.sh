#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== MCSR Ranked Scraping Campaign ==="
echo "Started: $(date -u)"
echo "Target: Top 2000 players, all ranked matches"
echo ""

python3 scraper/collect.py \
    --top-players 2000 \
    --output data/matches.jsonl \
    --checkpoint data/checkpoint.json \
    --resume \
    2>&1 | tee data/campaign.log

echo ""
echo "=== Campaign Complete ==="
echo "Finished: $(date -u)"
python3 scraper/stats.py data/matches.jsonl
