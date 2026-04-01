"""Configuration constants for the MCSR ranked prediction pipeline."""

from __future__ import annotations

SEED_OVERWORLD_TYPES = [
    "BURIED_TREASURE",
    "DESERT_TEMPLE",
    "RUINED_PORTAL",
    "SHIPWRECK",
    "VILLAGE",
]

SEED_NETHER_TYPES = ["BRIDGE", "HOUSING", "STABLES", "TREASURE"]

KEY_TIMELINE_EVENTS = [
    "story.enter_the_nether",
    "nether.find_bastion",
    "nether.find_fortress",
    "projectelo.timeline.blind_travel",
    "story.enter_the_end",
    "projectelo.timeline.dragon_death",
    "projectelo.timeline.death",
    "projectelo.timeline.death_spawnpoint",
    "projectelo.timeline.forfeit",
    "projectelo.timeline.reset",
]

DEFAULT_ELO = 1200.0
K_FACTOR = 32.0
MIN_MATCHES_FOR_SEED_ELO = 5
SHRINKAGE_FACTOR = 0.7

ROLLING_WINDOWS = [10, 20, 50]
SHORT_WINDOW_ELO_N = 20
VOLATILITY_WINDOW = 10

VALIDATION_FRACTION = 0.2
TEST_FRACTION = 0.2
RANDOM_STATE = 42
