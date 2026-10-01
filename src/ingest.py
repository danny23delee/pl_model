"""Download and cache raw football-data.co.uk CSVs, one per season."""
from __future__ import annotations

import datetime as dt
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | None = None) -> dict:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def season_code(start_year: int) -> str:
    """2000 -> '0001', 2025 -> '2526'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 7 else today.year - 1


def raw_path(cfg: dict, start_year: int) -> Path:
    return ROOT / cfg["data"]["raw_dir"] / f"{cfg['data']['league']}_{season_code(start_year)}.csv"


def download_season(cfg: dict, start_year: int, refresh: bool = False) -> Path | None:
    """Cache one season. Completed seasons are downloaded once; the current
    season is always refreshed. Returns None if the season does not exist yet."""
    dest = raw_path(cfg, start_year)
    is_current = start_year == current_season_start()
    if dest.exists() and not (refresh or is_current):
        return dest
    url = f"{cfg['data']['base_url']}/{season_code(start_year)}/{cfg['data']['league']}.csv"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pl-model-portfolio/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
    except Exception as e:  # noqa: BLE001
        if dest.exists():
            print(f"  {start_year}: download failed ({e}); using cache", file=sys.stderr)
            return dest
        print(f"  {start_year}: not available ({e})", file=sys.stderr)
        return None
    if len(body) < 200 or body.lstrip()[:1] == b"<":
        print(f"  {start_year}: response is not a CSV; skipping", file=sys.stderr)
        return dest if dest.exists() else None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return dest


def ingest(cfg: dict | None = None, refresh: bool = False) -> list[Path]:
    cfg = cfg or load_config()
    paths = []
    for y in range(cfg["data"]["first_season"], current_season_start() + 1):
        p = download_season(cfg, y, refresh)
        print(f"{y}/{(y + 1) % 100:02d}: {'ok' if p else 'missing'}")
        if p:
            paths.append(p)
    return paths


if __name__ == "__main__":
    ingest(refresh="--refresh" in sys.argv)
