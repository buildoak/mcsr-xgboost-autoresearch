#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                matches.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed JSON at line {line_no}")
    return matches


def ts_to_human(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def print_stats(matches: List[Dict[str, Any]]) -> None:
    total_matches = len(matches)
    print(f"Total matches: {total_matches}")

    if total_matches == 0:
        print("No matches found.")
        return

    date_values = [m.get("date") for m in matches if isinstance(m.get("date"), (int, float))]
    if date_values:
        min_date = int(min(date_values))
        max_date = int(max(date_values))
        print(f"Date range: {ts_to_human(min_date)} -> {ts_to_human(max_date)}")
    else:
        print("Date range: unavailable")

    unique_players = set()
    timelines_with = 0
    timelines_without = 0
    completion_times_sec: List[float] = []
    forfeited_count = 0
    seed_counter: Counter[str] = Counter()
    bastion_counter: Counter[str] = Counter()

    for match in matches:
        for player in match.get("players", []) or []:
            uuid = player.get("uuid")
            if uuid:
                unique_players.add(str(uuid))

        timelines = match.get("timelines")
        if isinstance(timelines, list) and len(timelines) > 0:
            timelines_with += 1
        else:
            timelines_without += 1

        completions = match.get("completions")
        if isinstance(completions, list):
            for c in completions:
                ms = c.get("time") if isinstance(c, dict) else None
                if isinstance(ms, (int, float)):
                    completion_times_sec.append(float(ms) / 1000.0)

        if bool(match.get("forfeited")):
            forfeited_count += 1

        seed_type = match.get("seedType")
        if seed_type is not None:
            seed_counter[str(seed_type)] += 1

        bastion_type = match.get("bastionType")
        if bastion_type is not None:
            bastion_counter[str(bastion_type)] += 1

    print(f"Unique players: {len(unique_players)}")
    print(f"Matches with timelines: {timelines_with}")
    print(f"Matches without timelines: {timelines_without}")

    if completion_times_sec:
        print(f"Average completion time: {mean(completion_times_sec):.2f} seconds")
    else:
        print("Average completion time: unavailable")

    forfeited_pct = (forfeited_count / total_matches) * 100.0
    print(f"Forfeited matches: {forfeited_count} ({forfeited_pct:.2f}%)")

    print("Top 5 seed types:")
    for seed_type, count in seed_counter.most_common(5):
        print(f"  {seed_type}: {count}")

    print("Top 5 bastion types:")
    for bastion_type, count in bastion_counter.most_common(5):
        print(f"  {bastion_type}: {count}")

    sample = matches[0]
    sample_out = {
        "id": sample.get("id"),
        "date": sample.get("date"),
        "date_human": ts_to_human(int(sample["date"])) if isinstance(sample.get("date"), (int, float)) else None,
        "players": [
            {
                "uuid": p.get("uuid"),
                "nickname": p.get("nickname"),
                "eloRate": p.get("eloRate"),
                "eloRank": p.get("eloRank"),
            }
            for p in (sample.get("players") or [])
            if isinstance(p, dict)
        ],
        "result": sample.get("result"),
        "forfeited": sample.get("forfeited"),
        "seedType": sample.get("seedType"),
        "bastionType": sample.get("bastionType"),
        "timelines_count": len(sample.get("timelines") or []),
        "completions": sample.get("completions"),
    }

    print("Sample match (first record):")
    print(json.dumps(sample_out, indent=2, ensure_ascii=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print summary statistics from MCSR match JSONL data.")
    parser.add_argument(
        "path",
        nargs="?",
        default="data/matches.jsonl",
        help="Path to matches JSONL file (default: data/matches.jsonl)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    input_path = Path(args.path)

    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")

    matches = load_jsonl(input_path)
    print_stats(matches)


if __name__ == "__main__":
    main()
