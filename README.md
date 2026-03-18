# MCSR Ranked: Predicting Minecraft Speedrun Matches with XGBoost

> **Work in progress.** Data collection ongoing, autoresearch loop not yet running. Current state: baseline pipeline with 0.8941 ROC-AUC on 43,455 Season 10 matches. Everything below documents the pipeline as it stands today -- the starting point, not the ceiling.

*XGBoost binary classifier for MCSR Ranked match outcomes, built as an autoresearch testbed.*

## Why Minecraft Speedrunning

Competitive Minecraft speedrunning is one of the only esports domains where the full game state timeline is recorded at millisecond precision. Tennis datasets give you outcomes: who won, what the score was, maybe serve speeds. Chess gives you moves. MCSR Ranked gives you 58 event types with ms timestamps for every match -- when each player entered the Nether, found the bastion, located the fortress, traveled blind to the stronghold, entered the End, killed the dragon. Per-player split times across six distinct game phases.

That timeline data creates two prediction problems stacked on top of each other. First: can you predict who wins a speedrun match *before it starts*, from player history and seed metadata alone? Second: can you predict the winner *during the match*, from live pace data as splits come in? No other competitive prediction dataset offers both.

This repo attacks the first problem. 43,455 matches. 85 features. 6 ELO systems. A dual-ELO architecture that tracks both the platform's official ratings and independent custom ratings with seed-type and bastion-type specialization. The baseline hits 0.8941 ROC-AUC on a strict temporal holdout -- strong enough to be interesting, early enough to have room.

The pipeline is also designed as a testbed for [autoresearch](https://x.com/karpathy/status/1886192184808149383) -- autonomous agent loops that iterate on feature engineering and model architecture. The companion project [tennis-xgboost-autoresearch](https://github.com/buildoak/tennis-xgboost-autoresearch) documents what happens when you let an optimizer iterate freely on a mutable codebase: +155 basis points of honest gain, then Goodhart's Law in live production. This repo will follow the same infrastructure once the data collection is complete.

## What This Is

An XGBoost binary classifier that predicts which player wins a given MCSR Ranked match. It processes 43,455 matches from Season 10, engineers 85 features from dual ELO systems, rolling player statistics, and Minecraft-specific timeline signals, and evaluates on a strict temporal holdout (80/20 time-split, no shuffling).

Not yet autoresearch. The autonomous loop infrastructure from the tennis project has not been ported. What exists today is a complete, tested prediction pipeline with a solid baseline and a data collection system designed to scale to the full season.

## Results

Baseline performance on 8,036 held-out test matches (the most recent 20% by date):

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.8941** |
| Accuracy | 80.5% |
| Feature count | 85 |
| Train matches | 32,140 |
| Test matches | 8,036 |
| Prediction mean | 0.494 |
| Prediction std | 0.337 |

This is the starting point, not the ceiling. The tennis project started at 0.7377 and reached 0.7611 through honest autoresearch iterations. MCSR's higher baseline (0.89 vs 0.74) likely means less room for automated improvement -- ELO already captures most of the signal. But the timeline feature space is barely explored.

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

Platform ELO difference dominates at 13.3% importance -- consistent with ELO being the strongest single predictor in any head-to-head competition. But positions 4-10 are all timeline-derived features: nether-to-end time, blind travel speed, end entry pace, completion time trends. This is the Minecraft-specific signal that pure ELO models miss.

## The ELO Trap

The MCSR Ranked API serves a subtle poison. The `players[].eloRate` field in each match record is the player's ELO **at scrape time**, not at match time. Use it directly as a feature and you get severe target leakage -- the rating already contains information from matches that haven't happened yet in your training timeline.

Pre-match ELO must be reconstructed from the `changes[]` array:

```
pre_match_elo = changes[].eloRate - changes[].change
```

`changes[].eloRate` is the **post-match** ELO. `changes[].change` is the delta applied after the match. Subtraction recovers the pre-match state. The pipeline handles this in `elo.py:PlatformELO.extract_pre_match_elos()`.

This trap is not documented in the API. If you are building a similar pipeline against MCSR Ranked data, this is the single most important implementation detail: the ELO field you see is not the ELO field you want.

## The Data

### What MCSR Ranked Is

[MCSR Ranked](https://mcsrranked.com/) is a competitive Minecraft speedrunning ladder. Two players queue into a 1v1 match on a shared random seed. Both start a fresh Minecraft world simultaneously. The goal: beat the Ender Dragon. The first player to complete the game wins. If one player forfeits, the other wins by default. The platform tracks ELO ratings, assigns ranks, and records full match timelines with event-level granularity.

Think chess ELO, but for Minecraft. Except unlike chess, the game state is partially random (seed generation), the execution space is physical (building, parkour, combat), and every match produces a millisecond-resolution timeline of game events.

### Source and Scope

Match data is collected from the [MCSR Ranked API](https://api.mcsrranked.com/) via the bundled scraper. Season 10 only (January 2 -- March 17, 2026). The scraper crawls the top 2,000 players by ELO, fetches their full ranked match histories, discovers additional opponents through match records, and hydrates each match with detailed metadata including timelines and seed information.

### Current Dataset

| Stat | Value |
|------|-------|
| Total matches | 43,455 |
| Unique players | 4,916 |
| Matches with timelines | 43,359 (99.8%) |
| Forfeited matches | 13,433 (30.9%) |
| Date range | Jan 2 -- Mar 17, 2026 |
| Storage | ~218 MB (JSONL, gitignored) |

The 30.9% forfeit rate is a domain signal, not noise. Forfeits happen when a player falls far enough behind that continuing is pointless -- bad Nether spawn, death to a blaze, failed blind travel. The model uses both career and rolling forfeit rates as features. Players who forfeit frequently are mechanically different from players who grind every match to completion.

Data is scraped locally and stored as newline-delimited JSON (`data/matches.jsonl`). Not committed to the repository -- run the scraper to build your own copy.

### Schema

Each match record contains:

| Field | Description |
|-------|-------------|
| `players[]` | Two player objects: UUID, nickname, ELO rate/rank, country |
| `result` | Winner UUID and completion time (ms) |
| `seed` | Overworld type, nether/bastion type, end tower heights, variations |
| `timelines[]` | Ordered game events: UUID, event type, timestamp (ms) |
| `changes[]` | Post-match ELO changes per player (UUID, change, eloRate) |
| `completions[]` | Completion records (UUID, time) for non-forfeited matches |
| `forfeited` | Boolean: did the match end by forfeit? |
| `date` | Unix timestamp |

## Feature Engineering

85 features across six categories. All features are computed chronologically -- the `build_dataset` loop extracts features from historical state, then updates state with the current match outcome. No future information leaks into any feature.

### Dual ELO System (12 features)

Two independent rating systems running in parallel:

- **Platform ELO.** Reconstructed pre-match ratings from the `changes[]` array (see The ELO Trap above). This is the official MCSR Ranked rating.
- **Custom ELO.** Independent K=32 tracking, provides a second signal uncorrelated with platform rating adjustments.

On top of the base ratings, four specialized variants:

- **Seed-type ELO.** Per-overworld-type ratings (Village, Shipwreck, Buried Treasure, Desert Temple, Ruined Portal) with Bayesian shrinkage toward global ELO when a player has fewer than 5 matches on a given seed type.
- **Bastion-type ELO.** Per-nether-type ratings (Bridge, Housing, Stables, Treasure) with the same shrinkage mechanism.
- **Short-window ELO.** Rolling window of the last 20 rating deltas, capturing recent form independent of career trajectory.
- **ELO volatility.** Standard deviation of recent rating changes, measuring consistency vs. streakiness.

All ELO differences are computed as interaction features: `elo_diff_platform`, `elo_diff_custom`, `elo_diff_seed`, `elo_diff_bastion`.

### Timeline-Derived Features (12 features)

Rolling historical averages per player for key game phase split times:

- **Nether entry time** -- how fast the player reaches the Nether
- **Bastion find time** -- navigation efficiency in the Nether
- **Fortress find time** -- blaze rod acquisition speed
- **Blind travel time** -- stronghold location phase
- **End entry time** -- portal room to End dimension
- **Nether-to-end time** -- derived: end entry minus nether entry, measures total Nether efficiency
- **Death rate** -- average deaths per match

These features capture playstyle and mechanical skill in ways that pure ELO cannot. A player with a 1500 ELO who reaches the Nether in 90 seconds is fundamentally different from a 1500 ELO player who takes 180 seconds but compensates with faster Nether navigation.

### Rolling Player Statistics (40+ features)

- **Win rates:** Career, last 10, last 20, last 50 matches
- **Forfeit rates:** Career and rolling 20-match window
- **Completion times:** Career average, last-20 average, standard deviation
- **Activity:** Matches in last 7/14/30 days, days since last match
- **Seed-specific win rates:** Per-overworld and per-bastion-type
- **Head-to-head:** Prior meetings and p1 win count between the specific matchup

### Context Features (10 features)

- **Seed features:** End tower heights (4 towers, mean, std), overworld/bastion type one-hot encoding, variation count
- **Match context:** Same country flag, hour of day, day of week

### Sparsity Guards

Seed-type and bastion-type ELO use Bayesian shrinkage: when a player has fewer than 5 matches on a given seed type, their seed-specific rating blends 70% seed-specific / 30% global. This prevents noisy estimates from dominating when sample sizes are small. The threshold (5 matches) and blend ratio (0.7) are configurable in `config.py`.

## The Timeline Edge

This is what makes MCSR different from every other match prediction dataset.

Tennis prediction has outcomes and surface types. Chess has move sequences. Basketball has box scores. MCSR Ranked has 58 distinct event types with millisecond timestamps, per player, per match. Not aggregated statistics after the fact -- raw event streams as the game unfolds.

A single completed MCSR match produces a timeline like:

```
Player A: enter_nether (92,341ms) -> find_bastion (141,220ms) -> find_fortress (198,445ms) -> blind_travel (312,891ms) -> enter_end (401,233ms) -> dragon_death (447,812ms)
Player B: enter_nether (88,102ms) -> find_bastion (155,891ms) -> death (178,003ms) -> find_fortress (221,445ms) -> blind_travel (345,221ms) -> forfeit
```

From this, the current pipeline extracts historical averages: "Player A's rolling average nether entry time is 95 seconds." But the raw timeline supports much more. Per-match split differentials. Pace modeling. In-match win probability that updates as each event fires. Phase-specific skill decomposition: fast overworld runner vs. efficient Nether navigator vs. clutch End fighter.

The baseline pipeline uses timeline data only as pre-match historical averages. The next phase -- live pace prediction -- would use in-match timeline events as they arrive to update win probability in real time. No other competitive esports dataset makes this possible at this granularity.

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

The scraper collects match data from the MCSR Ranked API. It is rate-limited (450 requests per 10-minute window) and fully checkpointed -- safe to interrupt and resume at any point.

```bash
# Collect matches from top 500 players (default)
python scraper/collect.py --resume

# Broader coverage: top 2000 players
python scraper/collect.py --top-players 2000 --resume

# Full campaign (used to build the current dataset)
bash campaign.sh
```

Output: `data/matches.jsonl` (one JSON record per line).

The scraper operates in three phases:

1. **Leaderboard fetch.** Get top N players by ELO from the MCSR Ranked API.
2. **Match history crawl.** Fetch all ranked matches for each player, discover opponents not in the original leaderboard, promote the highest-rated discoveries into the crawl queue.
3. **Detail hydration.** Fetch full match details (timelines, seed metadata, ELO changes) for all unique match IDs discovered in phase 2.

Checkpointing (`data/checkpoint.json`) tracks progress across all three phases. Use `--resume` to continue after interruption.

### B. Training Pipeline

```bash
python -m src.mcsr_predict.pipeline
```

This runs the full pipeline:

1. Load and filter matches (drops decayed matches, requires seed metadata)
2. Sort chronologically by date
3. Build 85 features with rolling state (features extracted before state update -- no leakage)
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

Prints summary statistics: match count, date range, unique players, timeline coverage, forfeit rate, seed type distribution, and a sample record.

### D. Running Tests

```bash
pytest tests/ -v
```

Tests cover ELO system correctness (pre-match extraction, update mechanics, shrinkage behavior) and feature engineering invariants (player ordering, temporal state isolation, rolling window calculations).

## Architecture

```
scraper/collect.py    Three-phase data collection
        |
        v
  data/matches.jsonl  Raw match records (JSONL, ~218MB, gitignored)
        |
        v
  pipeline.py         Load -> filter -> sort -> features -> split -> train -> evaluate
        |
        +---> features.py    FeatureBuilder: 85 features from ELO + timelines + rolling stats
        |         |
        |         +---> elo.py    Six ELO systems: platform, custom, seed, bastion, short-window, volatility
        |
        +---> models.py      XGBoost training with early stopping
        |
        +---> evaluate.py    ROC-AUC, accuracy, feature importance
        |
        +---> config.py      Constants: K-factors, windows, seed types, split ratio
```

**Design decisions:**

- **Chronological processing.** Features are computed before state is updated for each match. The `build_dataset` loop extracts features, then calls `_update_player_state`. This prevents target leakage from future match outcomes.
- **Deterministic player ordering.** Players are sorted by UUID to ensure consistent p1/p2 assignment regardless of API ordering.
- **Time-split evaluation.** Train/test split is purely temporal (first 80% / last 20%), no shuffling. This simulates real prediction conditions where you only have historical data.
- **Bayesian ELO shrinkage.** Seed-type and bastion-type ratings blend toward global ratings when match counts are low, preventing noisy estimates from sparse categories.

## Roadmap

What exists today is the baseline. What comes next:

1. **Complete data collection.** Season 10 is ongoing. The current 43,455 matches will grow to an estimated 140,000+ by season end. More data means denser player histories and more reliable timeline averages.
2. **Set up autoresearch loop.** Port the infrastructure from [tennis-xgboost-autoresearch](https://github.com/buildoak/tennis-xgboost-autoresearch): research program, verification gate with immutable evaluation, directive rotation, combat log, knowledge iteration detection.
3. **Feature space expansion.** With the full dataset: momentum features (win streak, recent form slope), player matchup style clustering, seed-specific completion time modeling, phase transition speed differentials.
4. **Live pace prediction.** A second model that takes in-match timeline events as they arrive and outputs real-time win probability. This is the unique opportunity that no other competitive dataset offers.

## Repository Structure

```
src/mcsr_predict/
  pipeline.py      End-to-end: load -> filter -> features -> train -> evaluate
  features.py      FeatureBuilder: 85 features from ELO, timelines, rolling stats
  elo.py           Six ELO systems: platform, custom, seed, bastion, short-window, volatility
  models.py        XGBoost training with early stopping
  evaluate.py      ROC-AUC, accuracy, feature importance
  config.py        Constants: K-factors, windows, seed types, split ratio

scraper/
  collect.py       Three-phase data collection with rate limiting and checkpointing
  stats.py         Dataset summary statistics

tests/
  test_elo.py        ELO system unit tests
  test_features.py   Feature engineering invariant tests

campaign.sh        Full scraping campaign script (top 2000 players)
requirements.txt   Python dependencies
```

## License

MIT for code (see `LICENSE`).

Match data is scraped from the public [MCSR Ranked API](https://api.mcsrranked.com/). Data files are not committed to this repository. Run the scraper to build your own dataset.

## Credits

- **[MCSR Ranked](https://mcsrranked.com/) / [Project Elo](https://mcsrranked.com/)** -- the competitive Minecraft speedrunning platform and API that makes this dataset possible
- **[speedrun.com](https://www.speedrun.com/)** -- the broader Minecraft speedrunning community and leaderboards
- **[Andrej Karpathy](https://x.com/karpathy/status/1886192184808149383)** -- the autoresearch pattern that this project is designed to run
- **[Tennis XGBoost Autoresearch](https://github.com/buildoak/tennis-xgboost-autoresearch)** -- the architectural predecessor: same pipeline pattern, same verification gate design, documented gaming taxonomy that informs this project's loop design
