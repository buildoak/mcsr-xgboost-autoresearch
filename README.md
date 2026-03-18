# MCSR Ranked: Predicting Minecraft Speedrun Matches with XGBoost

*XGBoost binary classifier for MCSR Ranked match outcomes, built as an autoresearch testbed.*

## What This Is

[MCSR Ranked](https://mcsrranked.com/) is a competitive Minecraft speedrunning ladder where players race 1v1 to beat the Ender Dragon. Think chess ELO but for Minecraft: each match starts on a shared random seed, both players try to complete the game as fast as possible, and the loser either finishes slower or forfeits. The platform tracks ELO ratings, match timelines (split times for each game phase), and detailed seed metadata.

This repo is an XGBoost binary classifier that predicts which player wins a given match. It processes 43,000+ matches from Season 10, engineers 85 features from dual ELO systems, rolling player statistics, and Minecraft-specific timeline signals, and achieves **0.8941 ROC-AUC** on a strict temporal holdout.

The prediction pipeline is also designed as a testbed for [autoresearch](https://x.com/karpathy/status/1886192184808149383) -- autonomous agent loops that iterate on feature engineering and model architecture. The companion project [tennis-xgboost-autoresearch](https://github.com/buildoak/tennis-xgboost-autoresearch) documents what happens when you let an optimizer iterate freely on a mutable codebase.

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.8941** |
| Accuracy | 80.5% |
| Feature count | 85 |
| Train matches | 32,140 |
| Test matches | 8,036 |
| Prediction distribution | mean=0.494, std=0.337 |

**Top 10 features by importance:**

| Feature | Importance |
|---------|------------|
| `elo_diff_platform` | 0.1333 |
| `platform_elo_p2` | 0.0464 |
| `platform_elo_p1` | 0.0394 |
| `completion_time_diff` | 0.0171 |
| `avg_nether_to_end_time_p2` | 0.0165 |
| `avg_end_entry_time_p2` | 0.0148 |
| `avg_blind_travel_time_p1` | 0.0146 |
| `avg_completion_time_last20_p1` | 0.0141 |
| `avg_end_entry_time_p1` | 0.0137 |
| `avg_blind_travel_time_p2` | 0.0126 |

Platform ELO difference is the strongest single predictor at 13.3% importance, consistent with ELO being the dominant signal in head-to-head competitions. Timeline-derived features (nether-to-end time, blind travel, end entry) collectively contribute significant signal -- this is the Minecraft-specific edge that generic ELO models miss.

## The Data

### Source

Match data is collected from the [MCSR Ranked API](https://api.mcsrranked.com/) via the bundled scraper. The scraper crawls the top 2,000 players by ELO, fetches their full ranked match histories, discovers additional opponents, and hydrates each match with detailed metadata.

### Current Dataset

- **43,455 match records** from Season 10 (Jan 2 -- Mar 17, 2026)
- **4,916 unique players**
- **43,359 matches with timeline data** (99.8%)
- **13,433 forfeited matches** (30.9%)
- Data is stored as newline-delimited JSON (`data/matches.jsonl`, ~218MB, gitignored)

### Schema

Each match record contains:

| Field | Description |
|-------|-------------|
| `players[]` | Two player objects with UUID, nickname, ELO rate/rank, country |
| `result` | Winner UUID and completion time (ms) |
| `seed` | Overworld type, nether/bastion type, end tower heights, variations |
| `timelines[]` | Ordered game events with UUID, event type, and timestamp |
| `changes[]` | Post-match ELO changes per player (`uuid`, `change`, `eloRate`) |
| `completions[]` | Completion records (UUID, time) for non-forfeited matches |
| `forfeited` | Boolean indicating whether the match ended by forfeit |
| `date` | Unix timestamp |

### The ELO Trap

The `players[].eloRate` field is the player's ELO **at scrape time**, not at match time. Using it directly as a feature is target leakage -- it contains information from future matches.

Pre-match ELO must be reconstructed from the `changes[]` array:

```
pre_match_elo = changes[].eloRate - changes[].change
```

This is because `changes[].eloRate` is the **post-match** ELO, and `changes[].change` is the delta. The pipeline handles this correctly in `elo.py:PlatformELO.extract_pre_match_elos()`.

## Feature Engineering

### Dual ELO System (12 features)

- **Platform ELO:** Reconstructed pre-match ratings from the `changes[]` array (see ELO trap above)
- **Custom ELO:** Independent tracking with K=32, provides a second signal uncorrelated with platform rating adjustments
- **Seed-type ELO:** Per-overworld-type ratings (Village, Shipwreck, etc.) with Bayesian shrinkage toward global ELO for sparse histories
- **Bastion-type ELO:** Per-nether-type ratings (Bridge, Housing, Stables, Treasure) with the same shrinkage mechanism
- **Short-window ELO:** Rolling window of last 20 rating deltas, capturing recent form
- **ELO volatility:** Standard deviation of recent rating changes, measuring consistency

All ELO differences are computed as features (`elo_diff_platform`, `elo_diff_custom`, `elo_diff_seed`, `elo_diff_bastion`).

### Timeline-Derived Features (12 features)

The unique edge. MCSR Ranked records split times for key game events:

- **Nether entry time** -- how fast the player reaches the Nether
- **Bastion find time** -- navigation efficiency in the Nether
- **Fortress find time** -- blaze rod acquisition speed
- **Blind travel time** -- stronghold location phase
- **End entry time** -- portal room to End dimension
- **Nether-to-end time** -- derived: end entry minus nether entry, measures Nether efficiency
- **Death rate** -- average deaths per match

Each is tracked as a rolling historical average per player. These features capture playstyle and mechanical skill in ways that pure ELO cannot.

### Rolling Player Statistics (40+ features)

- **Win rates:** Career, last 10, last 20, last 50 matches
- **Forfeit rates:** Career and rolling 20-match window
- **Completion times:** Career average, last-20 average, standard deviation
- **Activity:** Matches in last 7/14/30 days, days since last match
- **Seed-specific win rates:** Per-overworld and per-bastion-type
- **Head-to-head:** Prior meetings and p1 win count

### Context Features (10 features)

- **Seed features:** End tower heights (4 towers, mean, std), overworld/bastion type one-hot encoding, variation count
- **Match context:** Same country flag, hour of day, day of week

### Sparsity Guards

Seed-type and bastion-type ELO use Bayesian shrinkage: when a player has fewer than 5 matches on a given seed type, their seed-specific rating is blended with their global rating (70/30 split). This prevents noisy estimates from dominating when sample sizes are small.

## How to Replicate

### Prerequisites

Python >= 3.10, pip.

### Setup

```bash
git clone https://github.com/buildoak/mcsr-xgboost-autoresearch.git
cd mcsr-xgboost-autoresearch
pip install -r requirements.txt
```

### A. Data Collection

The scraper collects match data from the MCSR Ranked API. It is rate-limited and checkpointed -- safe to interrupt and resume.

```bash
# Collect matches from top 500 players (default)
python scraper/collect.py --resume

# Or target more players for broader coverage
python scraper/collect.py --top-players 2000 --resume

# Full campaign (used to build the current dataset)
bash campaign.sh
```

Output: `data/matches.jsonl` (one JSON record per line).

The scraper has three phases:
1. **Leaderboard fetch** -- get top N players by ELO
2. **Match history crawl** -- fetch all ranked matches for each player, discover opponents
3. **Detail hydration** -- fetch full match details (timelines, seed metadata) for all unique match IDs

Checkpointing (`data/checkpoint.json`) tracks progress across all three phases. Use `--resume` to continue after interruption.

### B. Training Pipeline

```bash
python -m src.mcsr_predict.pipeline
```

This runs the full pipeline:
1. Load and filter matches (drops decayed matches, requires seed metadata)
2. Sort chronologically
3. Build features (85 columns) with rolling state
4. Time-split: 80% train / 20% test (strict temporal, no shuffle)
5. Train XGBoost with early stopping (500 trees, depth 6, learning rate 0.05)
6. Evaluate and print metrics

Expected output:
```
Number of matches used: 40176
Train size: 32140
Test size: 8036
ROC-AUC: 0.8941
Accuracy: 0.8051
Feature count: 85
```

### C. Dataset Statistics

```bash
python scraper/stats.py
```

Prints summary statistics: match count, date range, unique players, timeline coverage, forfeit rate, and a sample record.

### D. Running Tests

```bash
pytest tests/ -v
```

Tests cover ELO system correctness (pre-match extraction, update mechanics, shrinkage behavior) and feature engineering invariants (player ordering, temporal state isolation, rolling window calculations).

## Architecture

```
src/mcsr_predict/
  pipeline.py    End-to-end: load -> filter -> features -> train -> evaluate
  features.py    FeatureBuilder: 85 features from ELO, timelines, rolling stats
  elo.py         Six ELO systems: platform, custom, seed, bastion, short-window, volatility
  models.py      XGBoost training with early stopping
  evaluate.py    ROC-AUC, accuracy, feature importance
  config.py      Constants: K-factors, windows, seed types, split ratio

scraper/
  collect.py     Three-phase data collection with rate limiting and checkpointing
  stats.py       Dataset summary statistics

tests/
  test_elo.py       ELO system unit tests
  test_features.py  Feature engineering invariant tests
```

**Design decisions:**
- **Chronological processing:** Features are computed before state is updated for each match. The `build_dataset` loop extracts features, then calls `_update_player_state`. This prevents target leakage from future match outcomes.
- **Deterministic player ordering:** Players are sorted by UUID to ensure consistent p1/p2 assignment regardless of API ordering.
- **Time-split evaluation:** Train/test split is purely temporal (first 80% / last 20%), no shuffling. This simulates real prediction conditions.

## Credits

- **[MCSR Ranked](https://mcsrranked.com/)** -- the competitive Minecraft speedrunning platform and API
- **[Andrej Karpathy](https://x.com/karpathy/status/1886192184808149383)** -- the autoresearch pattern

## License

MIT (see `LICENSE`)
