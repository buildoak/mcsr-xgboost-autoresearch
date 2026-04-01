"""Unit tests for feature engineering behavior."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mcsr_predict.features import FeatureBuilder


def make_match(
    *,
    date: int,
    uuid_a: str,
    uuid_b: str,
    winner_uuid: str,
    pre_elo_a: int = 1200,
    pre_elo_b: int = 1200,
    players_reversed: bool = False,
    forfeited: bool = False,
) -> dict:
    change_a = 10 if winner_uuid == uuid_a else -10
    change_b = 10 if winner_uuid == uuid_b else -10

    players = [
        {
            "uuid": uuid_a,
            "nickname": "a",
            "roleType": 3,
            "eloRate": pre_elo_a,
            "eloRank": 1,
            "country": "us",
        },
        {
            "uuid": uuid_b,
            "nickname": "b",
            "roleType": 3,
            "eloRate": pre_elo_b,
            "eloRank": 2,
            "country": "us",
        },
    ]
    if players_reversed:
        players = list(reversed(players))

    return {
        "id": date,
        "seed": {
            "id": "seed",
            "overworld": "VILLAGE",
            "nether": "TREASURE",
            "endTowers": [76, 82, 94, 97],
            "variations": ["biome:structure:deep_lukewarm_ocean"],
        },
        "players": players,
        "result": {"uuid": winner_uuid, "time": 600000},
        "forfeited": forfeited,
        "decayed": False,
        "changes": [
            {"uuid": uuid_a, "change": change_a, "eloRate": pre_elo_a + change_a},
            {"uuid": uuid_b, "change": change_b, "eloRate": pre_elo_b + change_b},
        ],
        "completions": [{"uuid": winner_uuid, "time": 600000}] if not forfeited else [],
        "timelines": [],
        "date": date,
    }


def test_player_ordering_is_alphabetical_uuid() -> None:
    builder = FeatureBuilder()

    match = make_match(
        date=100,
        uuid_a="a_uuid",
        uuid_b="b_uuid",
        winner_uuid="b_uuid",
        pre_elo_a=1200,
        pre_elo_b=1300,
        players_reversed=True,
    )

    X, y = builder.build_dataset([match])

    assert X.iloc[0]["platform_elo_p1"] == pytest.approx(1200.0)
    assert X.iloc[0]["platform_elo_p2"] == pytest.approx(1300.0)
    assert int(y.iloc[0]) == 0


def test_features_are_computed_before_state_update() -> None:
    builder = FeatureBuilder()

    m1 = make_match(date=100, uuid_a="a", uuid_b="b", winner_uuid="a")
    m2 = make_match(date=200, uuid_a="a", uuid_b="b", winner_uuid="b")

    X, _ = builder.build_dataset([m1, m2])

    assert X.iloc[0]["total_matches_played_p1"] == pytest.approx(0.0)
    assert X.iloc[0]["win_rate_career_p1"] == pytest.approx(0.0)

    assert X.iloc[1]["total_matches_played_p1"] == pytest.approx(1.0)
    assert X.iloc[1]["win_rate_career_p1"] == pytest.approx(1.0)
    assert X.iloc[1]["head_to_head_total"] == pytest.approx(1.0)
    assert X.iloc[1]["head_to_head_wins_p1"] == pytest.approx(1.0)


def test_rolling_window_features_with_known_results() -> None:
    builder = FeatureBuilder()

    day = 86400
    matches = [
        make_match(date=1 * day, uuid_a="a", uuid_b="b", winner_uuid="a"),
        make_match(date=2 * day, uuid_a="a", uuid_b="b", winner_uuid="b"),
        make_match(date=3 * day, uuid_a="a", uuid_b="b", winner_uuid="a"),
    ]

    X, _ = builder.build_dataset(matches)

    third_row = X.iloc[2]
    assert third_row["total_matches_played_p1"] == pytest.approx(2.0)
    assert third_row["win_rate_career_p1"] == pytest.approx(0.5)
    assert third_row["win_rate_last10_p1"] == pytest.approx(0.5)
    assert third_row["win_rate_last20_p1"] == pytest.approx(0.5)
    assert third_row["win_rate_last50_p1"] == pytest.approx(0.5)
    assert third_row["matches_last_7d_p1"] == pytest.approx(2.0)
    assert third_row["matches_last_14d_p1"] == pytest.approx(2.0)
    assert third_row["matches_last_30d_p1"] == pytest.approx(2.0)


def test_forfeited_match_with_reset_event_recovers_winner() -> None:
    builder = FeatureBuilder()
    match = make_match(date=100, uuid_a="a", uuid_b="b", winner_uuid="a", forfeited=True)
    match["result"]["uuid"] = None
    match["changes"] = [
        {"uuid": "a", "change": 0, "eloRate": 1200},
        {"uuid": "b", "change": 0, "eloRate": 1200},
    ]
    match["timelines"] = [{"uuid": "b", "time": 12345, "type": "projectelo.timeline.reset"}]

    X, y = builder.build_dataset([match])

    assert len(X) == 1
    assert int(y.iloc[0]) == 1
    assert builder.dataset_stats["rows_built"] == 1
    assert builder.dataset_stats["skipped_unresolved_winner"] == 0


def test_forfeited_match_with_positive_change_recovers_winner() -> None:
    builder = FeatureBuilder()
    match = make_match(date=100, uuid_a="a", uuid_b="b", winner_uuid="a", forfeited=True)
    match["result"]["uuid"] = None
    match["changes"] = [
        {"uuid": "a", "change": 12, "eloRate": 1212},
        {"uuid": "b", "change": -12, "eloRate": 1188},
    ]

    X, y = builder.build_dataset([match])

    assert len(X) == 1
    assert int(y.iloc[0]) == 1
