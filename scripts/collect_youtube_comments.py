"""Collect Khmer YouTube comments via the official YouTube Data API v3 (free).

Quota (free, 10,000 units/day per project): search.list = 100 units/call,
commentThreads.list = 1 unit/call (up to 100 comments per call).

Usage:
  python scripts/collect_youtube_comments.py --query "khmer food review" --query "khmer news" --max-videos 5
  python scripts/collect_youtube_comments.py --video-id D7UPBSnX --out data/raw/youtube_comments.csv
"""

import argparse
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.youtube import (
    API_KEY,
    SEARCH_UNITS,
    THREADS_UNITS,
    CollectBudget,
    QuotaExceededError,
    fetch_video_comments,
    get_json,
    search_videos,
)
from src.common.collect import write_collected_csv


def collect(queries, video_ids, max_videos, max_comments, budget_units, out):
    budget = CollectBudget(budget_units)
    session = requests.Session()
    all_rows = []

    for video_id in video_ids:
        all_rows.extend(
            fetch_video_comments(
                session, {"video_id": video_id, "title": "", "channel": ""}, max_comments, budget
            )
        )

    for query in queries:
        videos = search_videos(session, query, max_videos, budget)
        for video in videos:
            if budget.remaining() < THREADS_UNITS:
                print("Stopping early: budget nearly exhausted.")
                break
            all_rows.extend(fetch_video_comments(session, video, max_comments, budget))
        if budget.remaining() < SEARCH_UNITS:
            print("Stopping early: budget too low for another search.")
            break

    total = write_collected_csv(all_rows, out)
    print(f"Collected {total} unique comments -> {out}")
    print(f"Quota used: {budget.used} / {budget.budget} units")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", default=[], help="Khmer search term (repeatable)")
    parser.add_argument("--video-id", action="append", default=[], help="Specific video ID (repeatable)")
    parser.add_argument("--max-videos", type=int, default=10, help="Videos to search per query (default 10)")
    parser.add_argument("--max-comments", type=int, default=200, help="Comments per video (default 200)")
    parser.add_argument("--budget", type=int, default=1000, help="Quota units to spend (default 1000)")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "raw" / "youtube_comments.csv")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit("YOUTUBE_API_KEY is not set in .env - see .env.example")
    if not args.query and not args.video_id:
        raise SystemExit("Provide at least one --query or --video-id")

    collect(args.query, args.video_id, args.max_videos, args.max_comments, args.budget, args.out)


if __name__ == "__main__":
    main()