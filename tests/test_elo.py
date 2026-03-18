"""Unit tests for ELO systems."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mcsr_predict.elo import CustomELO, PlatformELO, SeedTypeELO


def _expected_delta(player_rating: float, opponent_rating: float, won: bool, k_factor: float = 32.0) -> float:
    expected = 1.0 / (1.0 + (10.0 ** ((opponent_rating - player_rating) / 400.0)))
    score = 1.0 if won else 0.0
    return k_factor * (score - expected)


def test_platform_elo_extracts_pre_match_from_changes() -> None:
    match = {
        "changes": [
            {"uuid": "p1", "change": 19, "eloRate": 2178},
            {"uuid": "p2", "change": -19, "eloRate": 2148},
        ]
    }

    platform = PlatformELO()
    ratings = platform.extract_pre_match_elos(match)

    assert ratings["p1"] == pytest.approx(2159.0)
    assert ratings["p2"] == pytest.approx(2167.0)


def test_custom_elo_updates_across_matches() -> None:
    elo = CustomELO(default_elo=1200.0, k_factor=32.0)

    p1, p2 = "a", "b"

    assert elo.get_rating(p1) == pytest.approx(1200.0)
    assert elo.get_rating(p2) == pytest.approx(1200.0)

    d1, d2 = elo.update_match(p1, p2, winner_uuid=p1)
    assert d1 == pytest.approx(16.0)
    assert d2 == pytest.approx(-16.0)

    r1 = 1200.0 + _expected_delta(1200.0, 1200.0, True)
    r2 = 1200.0 + _expected_delta(1200.0, 1200.0, False)
    assert elo.get_rating(p1) == pytest.approx(r1)
    assert elo.get_rating(p2) == pytest.approx(r2)

    old_r1, old_r2 = r1, r2
    elo.update_match(p1, p2, winner_uuid=p1)
    r1 = old_r1 + _expected_delta(old_r1, old_r2, True)
    r2 = old_r2 + _expected_delta(old_r2, old_r1, False)
    assert elo.get_rating(p1) == pytest.approx(r1)
    assert elo.get_rating(p2) == pytest.approx(r2)

    old_r1, old_r2 = r1, r2
    elo.update_match(p1, p2, winner_uuid=p2)
    d1_third = _expected_delta(old_r1, old_r2, False)
    d2_third = _expected_delta(old_r2, old_r1, True)
    assert elo.get_rating(p1) == pytest.approx(old_r1 + d1_third)
    assert elo.get_rating(p2) == pytest.approx(old_r2 + d2_third)


def test_seed_type_elo_applies_shrinkage_for_low_sample_players() -> None:
    global_elo = CustomELO(default_elo=1200.0)
    global_elo.ratings["p1"] = 1500.0

    seed_elo = SeedTypeELO(
        global_elo=global_elo,
        default_elo=1200.0,
        min_matches=5,
        shrinkage=0.7,
    )

    seed_elo.ratings["p1"]["VILLAGE"] = 1300.0
    seed_elo.match_counts["p1"]["VILLAGE"] = 3

    blended = seed_elo.get_rating("p1", "VILLAGE")
    assert blended == pytest.approx((0.7 * 1300.0) + (0.3 * 1500.0))

    seed_elo.match_counts["p1"]["VILLAGE"] = 5
    assert seed_elo.get_rating("p1", "VILLAGE") == pytest.approx(1300.0)
