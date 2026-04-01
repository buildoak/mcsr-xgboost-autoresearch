"""ELO systems used by feature extraction."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

from .config import (
    DEFAULT_ELO,
    K_FACTOR,
    MIN_MATCHES_FOR_SEED_ELO,
    SHORT_WINDOW_ELO_N,
    SHRINKAGE_FACTOR,
    VOLATILITY_WINDOW,
)


class PlatformELO:
    """Extracts platform pre-match ELO from match changes.

    Critical trap:
    `players[].eloRate` is scrape-time ELO and must NOT be used for historical matches.
    `changes[].eloRate` already contains the pre-match ELO for that player.
    """

    def __init__(self, default_elo: float = DEFAULT_ELO) -> None:
        self.default_elo = float(default_elo)

    def extract_pre_match_elos(self, match: dict[str, Any]) -> dict[str, float]:
        result: dict[str, float] = {}
        for entry in match.get("changes", []):
            uuid = entry.get("uuid")
            pre_elo = entry.get("eloRate")
            change = entry.get("change")
            if uuid is None or pre_elo is None or change is None:
                continue
            result[uuid] = float(pre_elo)
        return result

    def get_rating(self, match: dict[str, Any], player_uuid: str) -> float:
        return self.extract_pre_match_elos(match).get(player_uuid, self.default_elo)

    def get_ratings(self, match: dict[str, Any], p1_uuid: str, p2_uuid: str) -> tuple[float, float]:
        extracted = self.extract_pre_match_elos(match)
        return (
            extracted.get(p1_uuid, self.default_elo),
            extracted.get(p2_uuid, self.default_elo),
        )


class CustomELO:
    """Standard global ELO tracker."""

    def __init__(self, default_elo: float = DEFAULT_ELO, k_factor: float = K_FACTOR) -> None:
        self.default_elo = float(default_elo)
        self.k_factor = float(k_factor)
        self.ratings: defaultdict[str, float] = defaultdict(lambda: self.default_elo)

    @staticmethod
    def _expected_score(player_rating: float, opponent_rating: float) -> float:
        return 1.0 / (1.0 + (10.0 ** ((opponent_rating - player_rating) / 400.0)))

    def get_rating(self, player_uuid: str) -> float:
        return float(self.ratings[player_uuid])

    def get_ratings(self, p1_uuid: str, p2_uuid: str) -> tuple[float, float]:
        return self.get_rating(p1_uuid), self.get_rating(p2_uuid)

    def update_match(self, p1_uuid: str, p2_uuid: str, winner_uuid: str) -> tuple[float, float]:
        p1_rating = self.get_rating(p1_uuid)
        p2_rating = self.get_rating(p2_uuid)

        expected_p1 = self._expected_score(p1_rating, p2_rating)
        expected_p2 = self._expected_score(p2_rating, p1_rating)

        score_p1 = 1.0 if winner_uuid == p1_uuid else 0.0
        score_p2 = 1.0 - score_p1

        delta_p1 = self.k_factor * (score_p1 - expected_p1)
        delta_p2 = self.k_factor * (score_p2 - expected_p2)

        self.ratings[p1_uuid] = p1_rating + delta_p1
        self.ratings[p2_uuid] = p2_rating + delta_p2
        return float(delta_p1), float(delta_p2)


class SeedTypeELO:
    """Per-overworld ELO with shrinkage toward global ELO for sparse histories."""

    def __init__(
        self,
        global_elo: CustomELO,
        default_elo: float = DEFAULT_ELO,
        k_factor: float = K_FACTOR,
        min_matches: int = MIN_MATCHES_FOR_SEED_ELO,
        shrinkage: float = SHRINKAGE_FACTOR,
    ) -> None:
        self.global_elo = global_elo
        self.default_elo = float(default_elo)
        self.k_factor = float(k_factor)
        self.min_matches = int(min_matches)
        self.shrinkage = float(shrinkage)
        self.ratings: defaultdict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: self.default_elo)
        )
        self.match_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    @staticmethod
    def _expected_score(player_rating: float, opponent_rating: float) -> float:
        return 1.0 / (1.0 + (10.0 ** ((opponent_rating - player_rating) / 400.0)))

    def get_rating(self, player_uuid: str, seed_type: str | None) -> float:
        if not seed_type:
            return self.global_elo.get_rating(player_uuid)

        seed_rating = float(self.ratings[player_uuid][seed_type])
        seed_matches = int(self.match_counts[player_uuid][seed_type])
        global_rating = self.global_elo.get_rating(player_uuid)

        if seed_matches < self.min_matches:
            return (self.shrinkage * seed_rating) + ((1.0 - self.shrinkage) * global_rating)
        return seed_rating

    def get_ratings(self, p1_uuid: str, p2_uuid: str, seed_type: str | None) -> tuple[float, float]:
        return self.get_rating(p1_uuid, seed_type), self.get_rating(p2_uuid, seed_type)

    def update_match(self, p1_uuid: str, p2_uuid: str, winner_uuid: str, seed_type: str | None) -> None:
        if not seed_type:
            return

        p1_rating = float(self.ratings[p1_uuid][seed_type])
        p2_rating = float(self.ratings[p2_uuid][seed_type])

        expected_p1 = self._expected_score(p1_rating, p2_rating)
        expected_p2 = self._expected_score(p2_rating, p1_rating)

        score_p1 = 1.0 if winner_uuid == p1_uuid else 0.0
        score_p2 = 1.0 - score_p1

        self.ratings[p1_uuid][seed_type] = p1_rating + (self.k_factor * (score_p1 - expected_p1))
        self.ratings[p2_uuid][seed_type] = p2_rating + (self.k_factor * (score_p2 - expected_p2))

        self.match_counts[p1_uuid][seed_type] += 1
        self.match_counts[p2_uuid][seed_type] += 1


class BastionTypeELO(SeedTypeELO):
    """Per-nether/bastion-type ELO with global shrinkage for sparse samples."""


class ShortWindowELO:
    """ELO approximation where only the last N rating deltas are retained per player."""

    def __init__(
        self,
        window_size: int = SHORT_WINDOW_ELO_N,
        default_elo: float = DEFAULT_ELO,
        k_factor: float = K_FACTOR,
    ) -> None:
        self.window_size = int(window_size)
        self.default_elo = float(default_elo)
        self.k_factor = float(k_factor)
        self.recent_deltas: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    @staticmethod
    def _expected_score(player_rating: float, opponent_rating: float) -> float:
        return 1.0 / (1.0 + (10.0 ** ((opponent_rating - player_rating) / 400.0)))

    def get_rating(self, player_uuid: str) -> float:
        return self.default_elo + float(sum(self.recent_deltas[player_uuid]))

    def get_ratings(self, p1_uuid: str, p2_uuid: str) -> tuple[float, float]:
        return self.get_rating(p1_uuid), self.get_rating(p2_uuid)

    def update_match(self, p1_uuid: str, p2_uuid: str, winner_uuid: str) -> tuple[float, float]:
        p1_rating = self.get_rating(p1_uuid)
        p2_rating = self.get_rating(p2_uuid)

        expected_p1 = self._expected_score(p1_rating, p2_rating)
        expected_p2 = self._expected_score(p2_rating, p1_rating)

        score_p1 = 1.0 if winner_uuid == p1_uuid else 0.0
        score_p2 = 1.0 - score_p1

        delta_p1 = self.k_factor * (score_p1 - expected_p1)
        delta_p2 = self.k_factor * (score_p2 - expected_p2)

        self.recent_deltas[p1_uuid].append(float(delta_p1))
        self.recent_deltas[p2_uuid].append(float(delta_p2))
        return float(delta_p1), float(delta_p2)


class ELOVolatility:
    """Tracks rolling standard deviation of recent ELO deltas."""

    def __init__(self, window_size: int = VOLATILITY_WINDOW) -> None:
        self.window_size = int(window_size)
        self.delta_history: defaultdict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def get_volatility(self, player_uuid: str) -> float:
        history = self.delta_history[player_uuid]
        if len(history) < 2:
            return 0.0
        return float(np.std(np.array(history, dtype=float)))

    def get_volatilities(self, p1_uuid: str, p2_uuid: str) -> tuple[float, float]:
        return self.get_volatility(p1_uuid), self.get_volatility(p2_uuid)

    def update_match(self, p1_uuid: str, p2_uuid: str, p1_delta: float, p2_delta: float) -> None:
        self.delta_history[p1_uuid].append(float(p1_delta))
        self.delta_history[p2_uuid].append(float(p2_delta))
