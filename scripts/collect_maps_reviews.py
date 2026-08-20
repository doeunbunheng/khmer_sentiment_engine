"""Collect Google Maps reviews via the free Google Places API (legacy endpoints).

Free tier: Google Maps Platform gives $200/month credit on the billing account,
and a student-sized collector (a few searches + place details) stays within it.
Each Place Details call returns up to 5 reviews per place.

Usage:
  python scripts/collect_maps_reviews.py --query "restaurant Phnom Penh"
  python scripts/collect_maps_reviews.py --query "khmer cafe" --query "phnom penh restaurant" --max-places 3
  python scripts/collect_maps_reviews.py --place-id ChIJxxx --out data/raw/maps_reviews.csv
"""

import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.common.collect import maps_review_rows, write_collected_csv

API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
TEXTSEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"


def get_json(session, url, params):
    params = dict(params, key=API_KEY)
    resp = session.get(url, params=params, timeout=30)
    data = resp.json()
    status = data.get("status")
    if resp.status_code != 200 or status not in ("OK", "ZERO_RESULTS"):
        message = data.get("error_message") or status
        if "billing" in message.lower():
            message = (
                "Google Maps billing not enabled (free $200/month credit covers small use). "
                "Enable billing at https://console.cloud.google.com/project/_/billing/enable "
                "and the Places API at https://console.cloud.google.com/apis/library/places-backend.googleapis.com"
            )
        raise RuntimeError(f"Places API error {resp.status_code}: {message}")
    return data


def search_places(session, query, max_places):
    params = {"query": query}
    data = get_json(session, TEXTSEARCH_URL, params)
    return data.get("results", [])[:max_places]


def fetch_place_reviews(session, place):
    params = {"place_id": place["place_id"], "fields": "name,rating,reviews"}
    data = get_json(session, DETAILS_URL, params)
    result = data.get("result", {})
    result.setdefault("place_id", place["place_id"])
    result.setdefault("name", place.get("name", ""))
    return maps_review_rows(result, result.get("reviews"))


def collect(queries, place_ids, max_places, out):
    session = requests.Session()
    all_rows = []
    seen_places = set()

    for place_id in place_ids:
        if place_id in seen_places:
            continue
        seen_places.add(place_id)
        all_rows.extend(fetch_place_reviews(session, {"place_id": place_id, "name": ""}))

    for query in queries:
        places = search_places(session, query, max_places)
        for place in places:
            if place["place_id"] in seen_places:
                continue
            seen_places.add(place["place_id"])
            all_rows.extend(fetch_place_reviews(session, place))

    total = write_collected_csv(all_rows, out)
    print(f"Collected {total} unique reviews -> {out}")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", default=[], help="Place search term (repeatable)")
    parser.add_argument("--place-id", action="append", default=[], help="Specific place ID (repeatable)")
    parser.add_argument("--max-places", type=int, default=5, help="Places to fetch reviews for per query (default 5)")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "raw" / "maps_reviews.csv")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit("GOOGLE_PLACES_API_KEY is not set in .env - see .env.example")
    if not args.query and not args.place_id:
        raise SystemExit("Provide at least one --query or --place-id")

    collect(args.query, args.place_id, args.max_places, args.out)


if __name__ == "__main__":
    main()