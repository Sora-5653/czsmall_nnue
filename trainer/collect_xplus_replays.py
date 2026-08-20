#!/usr/bin/env python3
"""Collect a resumable corpus of X+ TETR.IO league replays.

The collection shape is intentionally similar to MochBot/fusion's public
recovered collector (cohort -> league records -> replay-id dedup -> replay
files), but this implementation is independent and rate-limit friendly:

* player/record discovery uses only the documented TETRA CHANNEL API;
* requests are serialized and throttled instead of proxy-rotated;
* pagination carries X-Session-ID as recommended by the API docs;
* replay bytes come from a configurable URL template because TETRA CHANNEL
  exposes replay ids, not a documented replay-download endpoint;
* every phase is append/resume friendly and produces machine-readable ledgers.

The default replay mirror is the public endpoint used by MochBot's recovered
collector. Override it with --replay-url-template (or
TETRA_REPLAY_URL_TEMPLATE) if a consented/private replay source is preferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

API_BASE_DEFAULT = "https://ch.tetr.io/api"
REPLAY_URL_TEMPLATE_DEFAULT = "https://inoue.szy.lol/api/replay/{replayid}"
USER_AGENT = "tetra-xplus-replay-collector/0.1 (research; TETR.IO league imitation pretraining)"


Json = dict[str, Any]


@dataclass(frozen=True)
class PlayerRef:
    user_id: str
    username: str | None
    rank: str
    tr: float | None
    country: str | None


@dataclass(frozen=True)
class ReplayRef:
    replay_id: str
    player_id: str
    rank: str
    timestamp: str | None
    record_id: str | None
    gamemode: str | None
    opponents: tuple[str, ...]


@dataclass(frozen=True)
class DownloadReport:
    replay_id: str
    path: str
    status: str
    bytes: int = 0
    sha256: str = ""
    error: str = ""


def _dict(value: Any) -> Json:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def normalize_rank(value: Any) -> str:
    return str(value or "").strip().lower()


def parse_rank_targets(values: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        rank = normalize_rank(value)
        if rank and rank not in out:
            out.append(rank)
    if not out:
        raise ValueError("at least one target rank is required")
    return tuple(out)


def cursor_from_entry(entry: Mapping[str, Any]) -> str | None:
    p = _dict(entry.get("p"))
    if not all(key in p for key in ("pri", "sec", "ter")):
        return None
    return f"{p['pri']}:{p['sec']}:{p['ter']}"


def player_from_entry(entry: Mapping[str, Any], targets: set[str]) -> PlayerRef | None:
    league = _dict(entry.get("league"))
    rank = normalize_rank(league.get("rank"))
    if rank not in targets:
        return None
    user_id = str(entry.get("_id") or "").strip()
    if not user_id:
        return None
    tr_raw = league.get("tr")
    try:
        tr = float(tr_raw) if tr_raw is not None else None
    except (TypeError, ValueError):
        tr = None
    username = entry.get("username")
    country = entry.get("country")
    return PlayerRef(
        user_id=user_id,
        username=str(username) if username is not None else None,
        rank=rank,
        tr=tr,
        country=str(country) if country is not None else None,
    )


def replay_ref_from_record(record: Mapping[str, Any], player: PlayerRef) -> ReplayRef | None:
    if record.get("stub"):
        return None
    replay_id = str(record.get("replayid") or "").strip()
    if not replay_id:
        return None
    opponents: list[str] = []
    for user in _list(record.get("otherusers")):
        uid = str(_dict(user).get("id") or "").strip()
        if uid:
            opponents.append(uid)
    return ReplayRef(
        replay_id=replay_id,
        player_id=player.user_id,
        rank=player.rank,
        timestamp=str(record.get("ts")) if record.get("ts") is not None else None,
        record_id=str(record.get("_id")) if record.get("_id") is not None else None,
        gamemode=str(record.get("gamemode")) if record.get("gamemode") is not None else None,
        opponents=tuple(opponents),
    )


def output_path_for(root: Path, ref: ReplayRef) -> Path:
    # IDs/ranks are server-originated but still sanitize separators so the
    # corpus layout cannot escape root if an upstream response is malformed.
    safe_rank = ref.rank.replace("/", "_").replace("\\", "_") or "unknown"
    safe_player = ref.player_id.replace("/", "_").replace("\\", "_")
    safe_replay = ref.replay_id.replace("/", "_").replace("\\", "_")
    return root / safe_rank / safe_player / f"{safe_replay}.ttrm"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[Json]:
    if not path.exists():
        return []
    out: list[Json] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def load_players(path: Path) -> list[PlayerRef]:
    players: list[PlayerRef] = []
    for item in read_jsonl(path):
        user_id = str(item.get("user_id") or item.get("_id") or "").strip()
        rank = normalize_rank(item.get("rank"))
        if not user_id or not rank:
            continue
        tr_raw = item.get("tr")
        try:
            tr = float(tr_raw) if tr_raw is not None else None
        except (TypeError, ValueError):
            tr = None
        players.append(PlayerRef(user_id, item.get("username"), rank, tr, item.get("country")))
    return players


def load_replay_refs(path: Path) -> list[ReplayRef]:
    refs: list[ReplayRef] = []
    seen: set[str] = set()
    for item in read_jsonl(path):
        replay_id = str(item.get("replay_id") or item.get("replayid") or "").strip()
        player_id = str(item.get("player_id") or "").strip()
        rank = normalize_rank(item.get("rank"))
        if not replay_id or not player_id or not rank or replay_id in seen:
            continue
        seen.add(replay_id)
        refs.append(
            ReplayRef(
                replay_id=replay_id,
                player_id=player_id,
                rank=rank,
                timestamp=item.get("timestamp") or item.get("ts"),
                record_id=item.get("record_id"),
                gamemode=item.get("gamemode"),
                opponents=tuple(str(v) for v in _list(item.get("opponents")) if v),
            )
        )
    return refs


def load_completed_player_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    for item in read_jsonl(path):
        player_id = str(item.get("player_id") or "").strip()
        if player_id and str(item.get("status") or "").lower() == "ok":
            completed.add(player_id)
    return completed


def load_downloaded_ids(path: Path) -> set[str]:
    done: set[str] = set()
    for item in read_jsonl(path):
        replay_id = str(item.get("replay_id") or item.get("replayid") or "").strip()
        status = str(item.get("status") or "").lower()
        if replay_id and status in {"ok", "exists", "200"}:
            done.add(replay_id)
    return done


class RateLimitedHttp:
    def __init__(self, *, interval: float, timeout: float, retries: int, user_agent: str = USER_AGENT):
        self.interval = max(0.0, interval)
        self.timeout = max(1.0, timeout)
        self.retries = max(0, retries)
        self.user_agent = user_agent
        self._last_started = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_started
        if self._last_started and elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_started = time.monotonic()

    def get(self, url: str, *, accept: str, session_id: str | None = None) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._throttle()
            headers = {"User-Agent": self.user_agent, "Accept": accept}
            if session_id:
                headers["X-Session-ID"] = session_id
            request = urllib.request.Request(url, headers=headers, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 429 and attempt < self.retries:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        wait = max(self.interval, float(retry_after)) if retry_after else max(2.0, self.interval)
                    except ValueError:
                        wait = max(2.0, self.interval)
                    time.sleep(wait)
                    continue
                if 500 <= exc.code < 600 and attempt < self.retries:
                    time.sleep(min(30.0, 2.0 ** attempt + random.random()))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(30.0, 2.0 ** attempt + random.random()))
                    continue
                raise
        raise RuntimeError(f"request failed: {url}: {last_error}")

    def get_json(self, url: str, *, session_id: str | None = None) -> Json:
        raw = self.get(url, accept="application/json", session_id=session_id)
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"JSON root is not an object: {url}")
        return value


def api_url(base: str, path: str, params: Mapping[str, Any]) -> str:
    base = base.rstrip("/")
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    return f"{base}/{path.lstrip('/')}" + (f"?{query}" if query else "")


def discover_players(
    http: RateLimitedHttp,
    *,
    api_base: str,
    targets: tuple[str, ...],
    max_pages: int,
    max_players: int,
) -> list[PlayerRef]:
    target_set = set(targets)
    players: list[PlayerRef] = []
    seen_ids: set[str] = set()
    cursor: str | None = None
    seen_target = False
    session_id = f"tetra-xplus-cohort-{int(time.time())}"

    for _page in range(max(1, max_pages)):
        payload = http.get_json(
            api_url(api_base, "users/by/league", {"limit": 100, "after": cursor}),
            session_id=session_id,
        )
        if not payload.get("success"):
            raise RuntimeError(f"leaderboard request failed: {payload.get('error')}")
        entries = _list(_dict(payload.get("data")).get("entries"))
        if not entries:
            break

        added = 0
        for raw in entries:
            entry = _dict(raw)
            player = player_from_entry(entry, target_set)
            if player is None or player.user_id in seen_ids:
                continue
            seen_target = True
            seen_ids.add(player.user_id)
            players.append(player)
            added += 1
            if max_players > 0 and len(players) >= max_players:
                return players

        cursor = cursor_from_entry(_dict(entries[-1]))
        if not cursor:
            break
        # League ranks occupy contiguous regions in the TR-sorted leaderboard.
        # Once the requested rank block has been seen and a full page contains
        # none of it, continuing would only walk lower ranks.
        if seen_target and added == 0:
            break
    return players


def fetch_player_records(
    http: RateLimitedHttp,
    *,
    api_base: str,
    player: PlayerRef,
    max_pages: int,
) -> list[ReplayRef]:
    refs: list[ReplayRef] = []
    cursor: str | None = None
    session_id = f"tetra-xplus-records-{player.user_id}-{int(time.time())}"
    for _page in range(max(1, max_pages)):
        payload = http.get_json(
            api_url(api_base, f"users/{urllib.parse.quote(player.user_id)}/records/league/recent", {"limit": 100, "after": cursor}),
            session_id=session_id,
        )
        if not payload.get("success"):
            raise RuntimeError(f"record request failed for {player.user_id}: {payload.get('error')}")
        entries = _list(_dict(payload.get("data")).get("entries"))
        if not entries:
            break
        for raw in entries:
            ref = replay_ref_from_record(_dict(raw), player)
            if ref is not None:
                refs.append(ref)
        cursor = cursor_from_entry(_dict(entries[-1]))
        if len(entries) < 100 or not cursor:
            break
    return refs


def validate_replay_bytes(raw: bytes) -> None:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("replay JSON root is not an object")
    # Keep validation deliberately permissive: ttrm_ingest owns the real shape
    # contract and supports canonical plus historical collector variants.
    if not any(key in value for key in ("data", "replay", "rounds")):
        raise ValueError("JSON does not look like a .ttrm replay document")


def download_replays(
    http: RateLimitedHttp,
    *,
    refs: Sequence[ReplayRef],
    root: Path,
    url_template: str,
    downloads_path: Path,
    max_replays: int,
    strict: bool,
) -> list[DownloadReport]:
    reports: list[DownloadReport] = []
    done = load_downloaded_ids(downloads_path)
    considered = 0
    for ref in refs:
        if max_replays > 0 and considered >= max_replays:
            break
        if ref.replay_id in done:
            continue
        considered += 1
        path = output_path_for(root, ref)
        if path.exists():
            report = DownloadReport(ref.replay_id, str(path), "exists", path.stat().st_size, "")
            append_jsonl(downloads_path, asdict(report))
            reports.append(report)
            done.add(ref.replay_id)
            continue

        url = url_template.format(replayid=urllib.parse.quote(ref.replay_id, safe=""))
        try:
            raw = http.get(url, accept="application/json,application/octet-stream")
            validate_replay_bytes(raw)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(raw)
            tmp.replace(path)
            report = DownloadReport(ref.replay_id, str(path), "ok", len(raw), sha256_bytes(raw))
            done.add(ref.replay_id)
        except Exception as exc:  # keep the corpus resumable after isolated mirror failures
            report = DownloadReport(ref.replay_id, str(path), "error", error=f"{type(exc).__name__}: {exc}")
            if strict:
                append_jsonl(downloads_path, asdict(report))
                raise
        append_jsonl(downloads_path, asdict(report))
        reports.append(report)
    return reports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=Path("data/xplus_replays"))
    ap.add_argument("--ranks", nargs="+", default=("x+",), help="target league ranks; default: x+")
    ap.add_argument("--api-base", default=API_BASE_DEFAULT)
    ap.add_argument(
        "--replay-url-template",
        default=os.environ.get("TETRA_REPLAY_URL_TEMPLATE", REPLAY_URL_TEMPLATE_DEFAULT),
        help="URL containing {replayid}; defaults to the public mirror used by MochBot's recovered collector",
    )
    ap.add_argument("--request-interval", type=float, default=1.05, help="minimum seconds between HTTP requests")
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--max-leaderboard-pages", type=int, default=200)
    ap.add_argument("--max-record-pages", type=int, default=50)
    ap.add_argument("--max-players", type=int, default=0, help="0 = no explicit cap")
    ap.add_argument("--max-replays", type=int, default=0, help="0 = download every indexed replay")
    ap.add_argument("--refresh-cohort", action="store_true")
    ap.add_argument("--refresh-records", action="store_true")
    ap.add_argument("--no-download", action="store_true", help="only build player/replay-id ledgers")
    ap.add_argument("--strict", action="store_true", help="abort on the first per-player/download error")
    args = ap.parse_args()

    if args.request_interval < 0 or args.timeout <= 0 or args.retries < 0:
        ap.error("request interval/retries must be non-negative and timeout must be positive")
    if args.max_leaderboard_pages <= 0 or args.max_record_pages <= 0:
        ap.error("page limits must be positive")
    targets = parse_rank_targets(args.ranks)

    root = args.output_dir.resolve()
    meta = root / "_meta"
    players_path = meta / "players.jsonl"
    replay_index_path = meta / "replay_index.jsonl"
    records_status_path = meta / "records_status.jsonl"
    downloads_path = meta / "downloads.jsonl"
    manifest_path = meta / "manifest.json"
    meta.mkdir(parents=True, exist_ok=True)

    http = RateLimitedHttp(interval=args.request_interval, timeout=args.timeout, retries=args.retries)

    players = [] if args.refresh_cohort else load_players(players_path)
    if players:
        players = [player for player in players if player.rank in set(targets)]
        print(f"cohort cache: {len(players)} players from {players_path}", flush=True)
    else:
        print(f"discovering ranks={','.join(targets)} via TETRA CHANNEL", flush=True)
        players = discover_players(
            http,
            api_base=args.api_base,
            targets=targets,
            max_pages=args.max_leaderboard_pages,
            max_players=args.max_players,
        )
        if not players:
            print("no target-rank players discovered", flush=True)
            return 1
        players_path.unlink(missing_ok=True)
        for player in players:
            append_jsonl(players_path, asdict(player))
        print(f"discovered {len(players)} target-rank players", flush=True)

    existing_refs = [] if args.refresh_records else load_replay_refs(replay_index_path)
    seen_replays = {ref.replay_id for ref in existing_refs}
    indexed_players = set() if args.refresh_records else load_completed_player_ids(records_status_path)
    # Backward compatibility for corpora created before records_status.jsonl:
    # any player with indexed replay ids was necessarily fetched successfully.
    indexed_players.update(ref.player_id for ref in existing_refs)
    if args.refresh_records:
        replay_index_path.unlink(missing_ok=True)
        records_status_path.unlink(missing_ok=True)
        existing_refs = []
        seen_replays.clear()
        indexed_players.clear()

    new_refs: list[ReplayRef] = []
    for index, player in enumerate(players, 1):
        if player.user_id in indexed_players:
            continue
        try:
            refs = fetch_player_records(
                http,
                api_base=args.api_base,
                player=player,
                max_pages=args.max_record_pages,
            )
        except Exception as exc:
            print(f"warning: records {player.username or player.user_id}: {exc}", flush=True)
            if args.strict:
                raise
            continue
        added = 0
        for ref in refs:
            if ref.replay_id in seen_replays:
                continue
            seen_replays.add(ref.replay_id)
            new_refs.append(ref)
            append_jsonl(replay_index_path, asdict(ref))
            added += 1
        append_jsonl(
            records_status_path,
            {
                "player_id": player.user_id,
                "status": "ok",
                "records_seen": len(refs),
                "new_unique_replays": added,
                "completed_at_unix": int(time.time()),
            },
        )
        indexed_players.add(player.user_id)
        if index % 10 == 0 or index == len(players):
            print(f"records {index}/{len(players)}; unique replays={len(seen_replays)} (+{added} last player)", flush=True)

    refs = load_replay_refs(replay_index_path)
    reports: list[DownloadReport] = []
    if not args.no_download:
        if "{replayid}" not in args.replay_url_template:
            ap.error("--replay-url-template must contain {replayid}")
        print(f"downloading up to {args.max_replays or len(refs)} indexed replays", flush=True)
        reports = download_replays(
            http,
            refs=refs,
            root=root,
            url_template=args.replay_url_template,
            downloads_path=downloads_path,
            max_replays=args.max_replays,
            strict=args.strict,
        )

    downloads = read_jsonl(downloads_path)
    ok_ids = {
        str(item.get("replay_id"))
        for item in downloads
        if str(item.get("status") or "").lower() in {"ok", "exists", "200"}
    }
    manifest = {
        "format": "tetra-xplus-replay-corpus-v1",
        "created_at_unix": int(time.time()),
        "target_ranks": list(targets),
        "api_base": args.api_base,
        "request_interval_seconds": args.request_interval,
        "players": len(players),
        "unique_replay_ids": len(refs),
        "downloaded_unique": len(ok_ids),
        "download_attempts_this_run": len(reports),
        "new_replay_ids_this_run": len(new_refs),
        "paths": {
            "players": str(players_path),
            "replay_index": str(replay_index_path),
            "records_status": str(records_status_path),
            "downloads": str(downloads_path),
        },
    }
    write_json(manifest_path, manifest)
    print(
        f"corpus ready: players={len(players)} replay_ids={len(refs)} downloaded={len(ok_ids)} manifest={manifest_path}",
        flush=True,
    )
    return 0 if (args.no_download or ok_ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
