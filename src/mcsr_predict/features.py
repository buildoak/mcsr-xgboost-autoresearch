"""Feature engineering for MCSR ranked match prediction."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from .config import KEY_TIMELINE_EVENTS, ROLLING_WINDOWS, SEED_NETHER_TYPES, SEED_OVERWORLD_TYPES
from .elo import BastionTypeELO, CustomELO, ELOVolatility, PlatformELO, SeedTypeELO, ShortWindowELO

EVENT_TO_AVG_FEATURE = {
    "story.enter_the_nether": "avg_nether_entry_time",
    "nether.find_bastion": "avg_bastion_find_time",
    "nether.find_fortress": "avg_fortress_find_time",
    "projectelo.timeline.blind_travel": "avg_blind_travel_time",
    "story.enter_the_end": "avg_end_entry_time",
}

DEATH_EVENTS = {"projectelo.timeline.death", "projectelo.timeline.death_spawnpoint"}


class FeatureBuilder:
    """Builds leakage-safe, chronological features for each match."""

    def __init__(self) -> None:
        self.platform_elo = PlatformELO()
        self.custom_elo = CustomELO()
        self.seed_elo = SeedTypeELO(global_elo=self.custom_elo)
        self.bastion_elo = BastionTypeELO(global_elo=self.custom_elo)
        self.short_window_elo = ShortWindowELO()
        self.elo_volatility = ELOVolatility()

        self.player_state: defaultdict[str, dict[str, Any]] = defaultdict(self._make_player_state)
        self.head_to_head: defaultdict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"wins_first": 0, "total": 0}
        )

    @staticmethod
    def _make_player_state() -> dict[str, Any]:
        return {
            "matches": 0,
            "wins": 0,
            "recent_wins": {window: deque(maxlen=window) for window in ROLLING_WINDOWS},
            "forfeits": 0,
            "recent_forfeits_20": deque(maxlen=20),
            "completion_sum": 0.0,
            "completion_sq_sum": 0.0,
            "completion_count": 0,
            "completion_recent_20": deque(maxlen=20),
            "completion_recent_20_sum": 0.0,
            "overworld_stats": defaultdict(lambda: {"wins": 0, "matches": 0}),
            "bastion_stats": defaultdict(lambda: {"wins": 0, "matches": 0}),
            "last_match_date": None,
            "recent_match_timestamps": deque(),
            "timeline_event_sum": defaultdict(float),
            "timeline_event_count": defaultdict(int),
            "nether_to_end_sum": 0.0,
            "nether_to_end_count": 0,
            "death_total": 0,
        }

    @staticmethod
    def _mean(values: deque[Any]) -> float:
        if not values:
            return 0.0
        return float(np.mean(np.array(values, dtype=float)))

    @staticmethod
    def _normalize_players(match: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
        players = match.get("players", [])
        if len(players) < 2:
            return None
        players_sorted = sorted(players[:2], key=lambda p: p.get("uuid", ""))
        return players_sorted[0], players_sorted[1]

    @staticmethod
    def _extract_seed_types(match: dict[str, Any]) -> tuple[str | None, str | None]:
        seed = match.get("seed") or {}
        return seed.get("overworld"), seed.get("nether")

    @staticmethod
    def _extract_end_towers(seed: dict[str, Any]) -> list[float]:
        towers = seed.get("endTowers") or []
        towers = [float(t) for t in towers[:4]]
        while len(towers) < 4:
            towers.append(0.0)
        towers.sort(reverse=True)
        return towers

    @staticmethod
    def _prune_recent_matches(state: dict[str, Any], current_ts: int) -> None:
        cutoff_30d = current_ts - (30 * 86400)
        recent = state["recent_match_timestamps"]
        while recent and recent[0] < cutoff_30d:
            recent.popleft()

    @staticmethod
    def _count_recent_matches(state: dict[str, Any], current_ts: int, days: int) -> int:
        cutoff = current_ts - (days * 86400)
        return int(sum(1 for ts in state["recent_match_timestamps"] if ts >= cutoff))

    def _player_historical_features(
        self, player_uuid: str, overworld_type: str | None, bastion_type: str | None, current_ts: int
    ) -> dict[str, float]:
        state = self.player_state[player_uuid]
        self._prune_recent_matches(state, current_ts)

        matches = int(state["matches"])
        wins = int(state["wins"])

        win_rate_career = (wins / matches) if matches else 0.0
        forfeit_rate_career = (state["forfeits"] / matches) if matches else 0.0

        completion_count = int(state["completion_count"])
        avg_completion_career = (state["completion_sum"] / completion_count) if completion_count else 0.0

        if completion_count >= 2:
            mean_completion = state["completion_sum"] / completion_count
            variance = max((state["completion_sq_sum"] / completion_count) - (mean_completion**2), 0.0)
            completion_std = sqrt(variance)
        else:
            completion_std = 0.0

        last_match_date = state["last_match_date"]
        days_since_last = ((current_ts - last_match_date) / 86400.0) if last_match_date is not None else -1.0

        overworld_stats = state["overworld_stats"][overworld_type] if overworld_type else {"wins": 0, "matches": 0}
        bastion_stats = state["bastion_stats"][bastion_type] if bastion_type else {"wins": 0, "matches": 0}

        overworld_rate = (
            overworld_stats["wins"] / overworld_stats["matches"] if overworld_stats["matches"] else 0.0
        )
        bastion_rate = bastion_stats["wins"] / bastion_stats["matches"] if bastion_stats["matches"] else 0.0

        timeline_avg: dict[str, float] = {}
        for event_name, feature_name in EVENT_TO_AVG_FEATURE.items():
            count = state["timeline_event_count"][event_name]
            timeline_avg[feature_name] = (
                state["timeline_event_sum"][event_name] / count if count else 0.0
            )

        nether_to_end_avg = (
            state["nether_to_end_sum"] / state["nether_to_end_count"] if state["nether_to_end_count"] else 0.0
        )
        death_rate = (state["death_total"] / matches) if matches else 0.0

        output = {
            "win_rate_career": win_rate_career,
            "win_rate_last10": self._mean(state["recent_wins"][10]),
            "win_rate_last20": self._mean(state["recent_wins"][20]),
            "win_rate_last50": self._mean(state["recent_wins"][50]),
            "forfeit_rate_career": forfeit_rate_career,
            "forfeit_rate_last20": self._mean(state["recent_forfeits_20"]),
            "avg_completion_time_career": avg_completion_career,
            "avg_completion_time_last20": (
                state["completion_recent_20_sum"] / len(state["completion_recent_20"])
                if state["completion_recent_20"]
                else 0.0
            ),
            "total_matches_played": float(matches),
            "win_rate_this_overworld": overworld_rate,
            "win_rate_this_bastion": bastion_rate,
            "days_since_last_match": days_since_last,
            "matches_last_7d": float(self._count_recent_matches(state, current_ts, 7)),
            "matches_last_14d": float(self._count_recent_matches(state, current_ts, 14)),
            "matches_last_30d": float(self._count_recent_matches(state, current_ts, 30)),
            "completion_time_std": completion_std,
            "death_rate": death_rate,
            "avg_nether_to_end_time": nether_to_end_avg,
        }
        output.update(timeline_avg)
        return output

    @staticmethod
    def _event_summary_by_player(match: dict[str, Any], player_uuids: set[str]) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
        per_player_events: dict[str, dict[str, float]] = {uuid: {} for uuid in player_uuids}
        per_player_deaths: dict[str, int] = {uuid: 0 for uuid in player_uuids}

        for event in match.get("timelines", []):
            uuid = event.get("uuid")
            if uuid not in player_uuids:
                continue

            event_type = event.get("type")
            event_time = event.get("time")
            if event_type is None or event_time is None:
                continue

            if event_type in KEY_TIMELINE_EVENTS and event_type in EVENT_TO_AVG_FEATURE:
                current = per_player_events[uuid].get(event_type)
                as_float = float(event_time)
                if current is None or as_float < current:
                    per_player_events[uuid][event_type] = as_float

            if event_type in DEATH_EVENTS:
                per_player_deaths[uuid] += 1

        return per_player_events, per_player_deaths

    def _update_player_state(
        self,
        player_uuid: str,
        won: bool,
        forfeited: bool,
        overworld_type: str | None,
        bastion_type: str | None,
        current_ts: int,
        completion_time: float | None,
        event_times: dict[str, float],
        deaths_in_match: int,
    ) -> None:
        state = self.player_state[player_uuid]

        state["matches"] += 1
        state["wins"] += int(won)

        for window in ROLLING_WINDOWS:
            state["recent_wins"][window].append(1 if won else 0)

        player_forfeit = int(forfeited and not won)
        state["forfeits"] += player_forfeit
        state["recent_forfeits_20"].append(player_forfeit)

        if completion_time is not None:
            state["completion_count"] += 1
            state["completion_sum"] += completion_time
            state["completion_sq_sum"] += completion_time**2

            recent_20 = state["completion_recent_20"]
            if len(recent_20) == recent_20.maxlen:
                state["completion_recent_20_sum"] -= recent_20[0]
            recent_20.append(completion_time)
            state["completion_recent_20_sum"] += completion_time

        if overworld_type:
            state["overworld_stats"][overworld_type]["matches"] += 1
            state["overworld_stats"][overworld_type]["wins"] += int(won)

        if bastion_type:
            state["bastion_stats"][bastion_type]["matches"] += 1
            state["bastion_stats"][bastion_type]["wins"] += int(won)

        recent_matches = state["recent_match_timestamps"]
        recent_matches.append(current_ts)
        self._prune_recent_matches(state, current_ts)

        state["last_match_date"] = current_ts

        for event_type, event_time in event_times.items():
            state["timeline_event_sum"][event_type] += float(event_time)
            state["timeline_event_count"][event_type] += 1

        if (
            "story.enter_the_nether" in event_times
            and "story.enter_the_end" in event_times
            and event_times["story.enter_the_end"] >= event_times["story.enter_the_nether"]
        ):
            state["nether_to_end_sum"] += (
                event_times["story.enter_the_end"] - event_times["story.enter_the_nether"]
            )
            state["nether_to_end_count"] += 1

        state["death_total"] += int(deaths_in_match)

    @staticmethod
    def _seed_features(seed: dict[str, Any]) -> dict[str, float]:
        overworld = seed.get("overworld")
        nether = seed.get("nether")

        towers = FeatureBuilder._extract_end_towers(seed)
        tower_mean = float(np.mean(np.array(towers, dtype=float)))
        tower_std = float(np.std(np.array(towers, dtype=float)))

        features: dict[str, float] = {
            "end_tower_1": towers[0],
            "end_tower_2": towers[1],
            "end_tower_3": towers[2],
            "end_tower_4": towers[3],
            "end_tower_mean": tower_mean,
            "end_tower_std": tower_std,
            "num_variations": float(len(seed.get("variations") or [])),
        }

        for seed_type in SEED_OVERWORLD_TYPES:
            features[f"overworld_type_{seed_type}"] = 1.0 if overworld == seed_type else 0.0
        for nether_type in SEED_NETHER_TYPES:
            features[f"bastion_type_{nether_type}"] = 1.0 if nether == nether_type else 0.0

        return features

    @staticmethod
    def _context_features(match: dict[str, Any], p1: dict[str, Any], p2: dict[str, Any]) -> dict[str, float]:
        p1_country = p1.get("country")
        p2_country = p2.get("country")

        same_country = int(
            p1_country is not None
            and p2_country is not None
            and str(p1_country).lower() == str(p2_country).lower()
        )

        ts = int(match.get("date", 0))
        dt = datetime.utcfromtimestamp(ts)

        return {
            "same_country": float(same_country),
            "hour_of_day": float(dt.hour),
            "day_of_week": float(dt.weekday()),
        }

    @staticmethod
    def _with_suffix(features: dict[str, float], suffix: str) -> dict[str, float]:
        return {f"{key}_{suffix}": float(value) for key, value in features.items()}

    def build_dataset(self, matches: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.Series]:
        rows: list[dict[str, float]] = []
        targets: list[int] = []

        for match in matches:
            seed = match.get("seed")
            if not seed:
                continue

            ordered = self._normalize_players(match)
            if ordered is None:
                continue

            p1, p2 = ordered
            p1_uuid = str(p1.get("uuid"))
            p2_uuid = str(p2.get("uuid"))
            if not p1_uuid or not p2_uuid:
                continue

            winner_uuid = str((match.get("result") or {}).get("uuid", ""))
            if winner_uuid not in {p1_uuid, p2_uuid}:
                continue

            current_ts = int(match.get("date", 0))
            overworld_type, bastion_type = self._extract_seed_types(match)

            platform_p1, platform_p2 = self.platform_elo.get_ratings(match, p1_uuid, p2_uuid)
            custom_p1, custom_p2 = self.custom_elo.get_ratings(p1_uuid, p2_uuid)
            seed_p1, seed_p2 = self.seed_elo.get_ratings(p1_uuid, p2_uuid, overworld_type)
            bastion_p1, bastion_p2 = self.bastion_elo.get_ratings(p1_uuid, p2_uuid, bastion_type)
            short_p1, short_p2 = self.short_window_elo.get_ratings(p1_uuid, p2_uuid)
            vol_p1, vol_p2 = self.elo_volatility.get_volatilities(p1_uuid, p2_uuid)

            p1_hist = self._player_historical_features(p1_uuid, overworld_type, bastion_type, current_ts)
            p2_hist = self._player_historical_features(p2_uuid, overworld_type, bastion_type, current_ts)

            h2h_key = (p1_uuid, p2_uuid)
            h2h_stats = self.head_to_head[h2h_key]

            row: dict[str, float] = {
                "platform_elo_p1": platform_p1,
                "platform_elo_p2": platform_p2,
                "custom_elo_p1": custom_p1,
                "custom_elo_p2": custom_p2,
                "seed_elo_p1": seed_p1,
                "seed_elo_p2": seed_p2,
                "bastion_elo_p1": bastion_p1,
                "bastion_elo_p2": bastion_p2,
                "short_window_elo_p1": short_p1,
                "short_window_elo_p2": short_p2,
                "elo_volatility_p1": vol_p1,
                "elo_volatility_p2": vol_p2,
                "elo_diff_platform": platform_p1 - platform_p2,
                "elo_diff_custom": custom_p1 - custom_p2,
                "elo_diff_seed": seed_p1 - seed_p2,
                "elo_diff_bastion": bastion_p1 - bastion_p2,
                "win_rate_diff_career": p1_hist["win_rate_career"] - p2_hist["win_rate_career"],
                "completion_time_diff": (
                    p1_hist["avg_completion_time_career"] - p2_hist["avg_completion_time_career"]
                ),
                "head_to_head_wins_p1": float(h2h_stats["wins_first"]),
                "head_to_head_total": float(h2h_stats["total"]),
            }
            row.update(self._with_suffix(p1_hist, "p1"))
            row.update(self._with_suffix(p2_hist, "p2"))
            row.update(self._seed_features(seed))
            row.update(self._context_features(match, p1, p2))

            rows.append(row)
            targets.append(1 if winner_uuid == p1_uuid else 0)

            # Update all rolling state only after current features are materialized.
            custom_delta_p1, custom_delta_p2 = self.custom_elo.update_match(p1_uuid, p2_uuid, winner_uuid)
            self.seed_elo.update_match(p1_uuid, p2_uuid, winner_uuid, overworld_type)
            self.bastion_elo.update_match(p1_uuid, p2_uuid, winner_uuid, bastion_type)
            self.short_window_elo.update_match(p1_uuid, p2_uuid, winner_uuid)
            self.elo_volatility.update_match(p1_uuid, p2_uuid, custom_delta_p1, custom_delta_p2)

            h2h_stats["total"] += 1
            if winner_uuid == p1_uuid:
                h2h_stats["wins_first"] += 1

            forfeited = bool(match.get("forfeited", False))
            completion_time = (match.get("result") or {}).get("time")
            winner_completion = float(completion_time) if (completion_time is not None and not forfeited) else None

            event_times, death_counts = self._event_summary_by_player(match, {p1_uuid, p2_uuid})

            self._update_player_state(
                player_uuid=p1_uuid,
                won=(winner_uuid == p1_uuid),
                forfeited=forfeited,
                overworld_type=overworld_type,
                bastion_type=bastion_type,
                current_ts=current_ts,
                completion_time=winner_completion if winner_uuid == p1_uuid else None,
                event_times=event_times[p1_uuid],
                deaths_in_match=death_counts[p1_uuid],
            )
            self._update_player_state(
                player_uuid=p2_uuid,
                won=(winner_uuid == p2_uuid),
                forfeited=forfeited,
                overworld_type=overworld_type,
                bastion_type=bastion_type,
                current_ts=current_ts,
                completion_time=winner_completion if winner_uuid == p2_uuid else None,
                event_times=event_times[p2_uuid],
                deaths_in_match=death_counts[p2_uuid],
            )

        feature_df = pd.DataFrame(rows)
        if not feature_df.empty:
            feature_df = feature_df.fillna(0.0)
        target_series = pd.Series(targets, dtype=int, name="target")
        return feature_df, target_series
