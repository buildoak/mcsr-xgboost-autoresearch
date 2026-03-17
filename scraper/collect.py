#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
from requests import Response
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout
from tqdm import tqdm

BASE_URL = "https://api.mcsrranked.com"
RATE_LIMIT_REQUESTS = 450
RATE_LIMIT_WINDOW_SECONDS = 600.0
RETRY_BACKOFF_SECONDS = (5, 15, 45)
REQUEST_TIMEOUT_SECONDS = 30
PHASE3_CHECKPOINT_INTERVAL = 50

logger = logging.getLogger(__name__)


@dataclass
class Player:
    uuid: str
    nickname: str
    elo_rate: int


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_timestamps: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            now = time.monotonic()
            while self.request_timestamps and now - self.request_timestamps[0] >= self.window_seconds:
                self.request_timestamps.popleft()

            if len(self.request_timestamps) < self.max_requests:
                self.request_timestamps.append(now)
                return

            sleep_for = self.window_seconds - (now - self.request_timestamps[0])
            if sleep_for > 0:
                logger.info("Rate limit window full. Sleeping %.2f seconds.", sleep_for)
                time.sleep(sleep_for)


class MCSRClient:
    def __init__(self, base_url: str, rate_limiter: SlidingWindowRateLimiter) -> None:
        self.base_url = base_url.rstrip("/")
        self.rate_limiter = rate_limiter
        self.session = requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        retries: int = 3,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        attempts = retries + 1

        for attempt in range(attempts):
            self.rate_limiter.acquire()
            try:
                response: Response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except (RequestsConnectionError, Timeout) as exc:
                if attempt < retries:
                    backoff = RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
                    logger.warning(
                        "Network error on %s %s (%s). Retry %d/%d in %ds.",
                        method,
                        url,
                        exc,
                        attempt + 1,
                        retries,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise RuntimeError(f"Network request failed after retries: {method} {url}") from exc

            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt < retries:
                    backoff = RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
                    logger.warning(
                        "Retryable HTTP %d for %s %s. Retry %d/%d in %ds.",
                        response.status_code,
                        method,
                        url,
                        attempt + 1,
                        retries,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise RuntimeError(
                    f"Request failed after retries with HTTP {response.status_code}: {method} {url}"
                )

            if response.status_code >= 400:
                raise RuntimeError(
                    f"Non-retryable HTTP {response.status_code} for {method} {url}: {response.text[:300]}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise RuntimeError(f"Invalid JSON response for {method} {url}") from exc

            if payload.get("status") != "success":
                raise RuntimeError(f"API returned non-success status for {method} {url}: {payload}")

            data = payload.get("data")
            return data

        raise RuntimeError(f"Unreachable: exhausted attempts for {method} {url}")

    def fetch_leaderboard(self, count: int = 150) -> Dict[str, Any]:
        return self._request("GET", "/leaderboard", params={"count": count})

    def fetch_user_matches(self, uuid: str, page: int, count: int = 100, match_type: int = 2) -> List[Dict[str, Any]]:
        data = self._request(
            "GET",
            f"/users/{uuid}/matches",
            params={"count": count, "page": page, "type": match_type},
        )
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected match history response for user {uuid}: {type(data)}")
        return data

    def fetch_match_detail(self, match_id: str) -> Dict[str, Any]:
        data = self._request("GET", f"/matches/{match_id}")
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected match detail response for match {match_id}: {type(data)}")
        return data


class MatchJsonlWriter:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_path.open("a", encoding="utf-8")

    def write(self, match_detail: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(match_detail, separators=(",", ":")) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class CheckpointStore:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path

    def load(self) -> Dict[str, Any]:
        if not self.checkpoint_path.exists():
            return {
                "completed_players": set(),
                "seen_match_ids": set(),
                "discovered_players": {},
            }

        with self.checkpoint_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        completed_players = set(str(u) for u in raw.get("completed_players", []))
        seen_match_ids = set(str(mid) for mid in raw.get("seen_match_ids", []))

        discovered_players: Dict[str, Player] = {}
        for entry in raw.get("discovered_players", []):
            if not isinstance(entry, dict):
                continue
            uuid = str(entry.get("uuid", "")).strip()
            if not uuid:
                continue
            discovered_players[uuid] = Player(
                uuid=uuid,
                nickname=str(entry.get("nickname") or uuid),
                elo_rate=_safe_int(entry.get("eloRate"), 0),
            )

        return {
            "completed_players": completed_players,
            "seen_match_ids": seen_match_ids,
            "discovered_players": discovered_players,
        }

    def save(
        self,
        completed_players: Set[str],
        seen_match_ids: Set[str],
        discovered_players: Dict[str, Player],
    ) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")

        discovered_list = [
            {"uuid": p.uuid, "nickname": p.nickname, "eloRate": p.elo_rate}
            for p in sorted(discovered_players.values(), key=lambda x: x.elo_rate, reverse=True)
        ]

        payload = {
            "completed_players": sorted(completed_players),
            "seen_match_ids": sorted(seen_match_ids, key=_sortable_match_id),
            "discovered_players": discovered_list,
        }

        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=True)
            fh.flush()
            os.fsync(fh.fileno())

        os.replace(tmp_path, self.checkpoint_path)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sortable_match_id(value: str) -> Tuple[int, Any]:
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def load_existing_output_match_ids(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()

    match_ids: Set[str] = set()
    with output_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line %d in %s", line_no, output_path)
                continue
            match_id = record.get("id")
            if match_id is None:
                continue
            match_ids.add(str(match_id))

    return match_ids


def parse_players_from_leaderboard(leaderboard_data: Dict[str, Any], target_count: int) -> List[Player]:
    users = leaderboard_data.get("users", [])
    players: List[Player] = []
    for user in users:
        uuid = str(user.get("uuid", "")).strip()
        if not uuid:
            continue
        players.append(
            Player(
                uuid=uuid,
                nickname=str(user.get("nickname") or uuid),
                elo_rate=_safe_int(user.get("eloRate"), 0),
            )
        )
        if len(players) >= target_count:
            break
    return players


def promote_discovered_players(
    discovered_pool: Dict[str, Player],
    player_map: Dict[str, Player],
    player_queue: List[Player],
    top_players: int,
) -> int:
    if len(player_map) >= top_players:
        return 0

    added = 0
    candidates = sorted(discovered_pool.values(), key=lambda p: p.elo_rate, reverse=True)
    for candidate in candidates:
        if len(player_map) >= top_players:
            break
        if candidate.uuid in player_map:
            discovered_pool.pop(candidate.uuid, None)
            continue
        player_map[candidate.uuid] = candidate
        player_queue.append(candidate)
        discovered_pool.pop(candidate.uuid, None)
        added += 1

    return added


def discover_players_from_matches(
    matches: Iterable[Dict[str, Any]],
    player_map: Dict[str, Player],
    discovered_pool: Dict[str, Player],
) -> int:
    discovered = 0
    for match in matches:
        for player in match.get("players", []) or []:
            uuid = str(player.get("uuid", "")).strip()
            if not uuid or uuid in player_map:
                continue
            nickname = str(player.get("nickname") or uuid)
            elo_rate = _safe_int(player.get("eloRate"), 0)

            existing = discovered_pool.get(uuid)
            if existing is None:
                discovered_pool[uuid] = Player(uuid=uuid, nickname=nickname, elo_rate=elo_rate)
                discovered += 1
            elif elo_rate > existing.elo_rate:
                discovered_pool[uuid] = Player(uuid=uuid, nickname=nickname, elo_rate=elo_rate)

    return discovered


def fetch_all_ranked_matches_for_player(
    client: MCSRClient, player_uuid: str, max_matches: int = 0
) -> List[Dict[str, Any]]:
    all_matches: List[Dict[str, Any]] = []
    page = 0
    while True:
        page_matches = client.fetch_user_matches(player_uuid, page=page, count=100, match_type=2)
        if not page_matches:
            break
        all_matches.extend(page_matches)
        if max_matches > 0 and len(all_matches) >= max_matches:
            all_matches = all_matches[:max_matches]
            break
        page += 1
    return all_matches


def run_collection(top_players: int, output_path: Path, checkpoint_path: Path, resume: bool, max_matches_per_player: int = 0) -> None:
    client = MCSRClient(
        base_url=BASE_URL,
        rate_limiter=SlidingWindowRateLimiter(
            max_requests=RATE_LIMIT_REQUESTS,
            window_seconds=RATE_LIMIT_WINDOW_SECONDS,
        ),
    )

    checkpoint_store = CheckpointStore(checkpoint_path)
    if resume:
        checkpoint = checkpoint_store.load()
        completed_players: Set[str] = checkpoint["completed_players"]
        seen_match_ids: Set[str] = checkpoint["seen_match_ids"]
        discovered_pool: Dict[str, Player] = checkpoint["discovered_players"]
        logger.info(
            "Loaded checkpoint: %d completed players, %d seen match IDs, %d discovered players.",
            len(completed_players),
            len(seen_match_ids),
            len(discovered_pool),
        )
    else:
        completed_players = set()
        seen_match_ids = set()
        discovered_pool = {}

    # Phase 1: Build base player list from leaderboard (max API-supported 150 in one call)
    leaderboard_count = min(top_players, 150)
    leaderboard_data = client.fetch_leaderboard(count=leaderboard_count)
    leaderboard_players = parse_players_from_leaderboard(leaderboard_data, leaderboard_count)

    player_map: Dict[str, Player] = {p.uuid: p for p in leaderboard_players}
    player_queue: List[Player] = list(leaderboard_players)

    if top_players > 150 and discovered_pool:
        added = promote_discovered_players(discovered_pool, player_map, player_queue, top_players)
        if added:
            logger.info("Restored %d players from discovered pool for resume.", added)

    logger.info("Phase 1 complete: %d/%d target players currently known.", len(player_queue), top_players)

    # Phase 2: Collect full ranked match history for each target player and discover additional players.
    initial_progress = min(sum(1 for p in player_queue if p.uuid in completed_players), top_players)
    with tqdm(total=top_players, initial=initial_progress, desc="Players", unit="player") as player_pbar:
        idx = 0
        while idx < len(player_queue):
            player = player_queue[idx]
            idx += 1

            if player.uuid in completed_players:
                continue

            player_index = min(idx, top_players)
            logger.info("Processing player %d/%d: %s (%s)", player_index, top_players, player.nickname, player.uuid)

            try:
                player_matches = fetch_all_ranked_matches_for_player(client, player.uuid, max_matches=max_matches_per_player)
            except Exception as exc:
                logger.exception("Failed to fetch match history for player %s: %s", player.uuid, exc)
                player_pbar.update(1)
                checkpoint_store.save(completed_players, seen_match_ids, discovered_pool)
                continue

            found_count = len(player_matches)
            new_match_count = 0
            for match in player_matches:
                match_id = match.get("id")
                if match_id is None:
                    continue
                key = str(match_id)
                if key not in seen_match_ids:
                    seen_match_ids.add(key)
                    new_match_count += 1

            discovered_count = 0
            if len(player_map) < top_players:
                discovered_count = discover_players_from_matches(player_matches, player_map, discovered_pool)
                promoted = promote_discovered_players(discovered_pool, player_map, player_queue, top_players)
                if promoted:
                    logger.info(
                        "Promoted %d discovered players (known players: %d/%d).",
                        promoted,
                        len(player_queue),
                        top_players,
                    )

            logger.info(
                "Player %s: matches found=%d, unique new matches=%d, newly discovered players=%d",
                player.uuid,
                found_count,
                new_match_count,
                discovered_count,
            )

            completed_players.add(player.uuid)
            player_pbar.update(1)
            checkpoint_store.save(completed_players, seen_match_ids, discovered_pool)

            if len(player_queue) >= top_players and idx >= top_players:
                break

    if len(player_queue) < top_players:
        logger.warning(
            "Only %d players available after discovery, below requested top_players=%d.",
            len(player_queue),
            top_players,
        )

    logger.info(
        "Phase 2 complete: %d completed players, %d unique match IDs collected.",
        len(completed_players),
        len(seen_match_ids),
    )

    # Phase 3: Fetch details for all unique unseen match IDs and append to JSONL.
    already_written_ids = load_existing_output_match_ids(output_path)
    pending_match_ids = [mid for mid in seen_match_ids if mid not in already_written_ids]
    logger.info(
        "Phase 3 starting: %d total IDs, %d already in output, %d pending detail fetches.",
        len(seen_match_ids),
        len(already_written_ids),
        len(pending_match_ids),
    )

    writer = MatchJsonlWriter(output_path)
    try:
        with tqdm(total=len(pending_match_ids), desc="Match details", unit="match") as match_pbar:
            for idx, match_id in enumerate(pending_match_ids, start=1):
                try:
                    match_detail = client.fetch_match_detail(match_id)
                    writer.write(match_detail)
                except Exception as exc:
                    logger.exception("Failed to fetch match detail for %s. Skipping. Error: %s", match_id, exc)
                finally:
                    match_pbar.update(1)

                if idx % PHASE3_CHECKPOINT_INTERVAL == 0:
                    checkpoint_store.save(completed_players, seen_match_ids, discovered_pool)

        checkpoint_store.save(completed_players, seen_match_ids, discovered_pool)
    finally:
        writer.close()

    logger.info("Collection complete. Output written to %s", output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect MCSR Ranked match data for modeling.")
    parser.add_argument("--top-players", type=int, default=500, help="Target number of players to scrape.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/matches.jsonl"),
        help="Output JSONL path for match detail records.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/checkpoint.json"),
        help="Checkpoint JSON path for resume state.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint and existing output.")
    parser.add_argument(
        "--max-matches-per-player",
        type=int,
        default=0,
        help="Max matches to fetch per player (0 = unlimited). Useful for initial testing.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.top_players <= 0:
        raise SystemExit("--top-players must be > 0")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_collection(
        top_players=args.top_players,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        max_matches_per_player=args.max_matches_per_player,
    )


if __name__ == "__main__":
    main()
