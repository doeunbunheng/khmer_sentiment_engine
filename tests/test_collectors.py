"""Tests for data-collection helpers (no network calls)."""

from pathlib import Path

import pandas as pd

from src.collectors.youtube import CollectBudget, QuotaExceededError, extract_video_id
from src.common.collect import maps_review_rows, write_collected_csv, youtube_comment_rows


def _read_df(path):
    return pd.read_csv(path, encoding="utf-8")


def test_write_collected_csv_utf8_and_text_first(tmp_path):
    out = tmp_path / "out.csv"
    rows = [
        {"text": "អាហារឆ្ងាញ់", "source": "youtube:abc", "author": "a"},
        {"text": "good product", "source": "maps:xyz", "author": "b"},
    ]
    n = write_collected_csv(rows, out)
    assert n == 2
    df = _read_df(out)
    assert list(df.columns)[0] == "text"
    assert set(df.columns) == {"text", "source", "author"}
    assert df.iloc[0]["text"] == "អាហារឆ្ងាញ់"


def test_write_collected_csv_dedupes_and_drops_blank(tmp_path):
    out = tmp_path / "out.csv"
    rows = [
        {"text": "same text"},
        {"text": "same text"},
        {"text": "   "},
        {"text": None},
        {"text": "another"},
    ]
    n = write_collected_csv(rows, out)
    assert n == 2
    assert list(_read_df(out)["text"]) == ["same text", "another"]


def test_write_collected_csv_empty(tmp_path):
    out = tmp_path / "out.csv"
    n = write_collected_csv([], out)
    assert n == 0
    df = _read_df(out)
    assert "text" in df.columns
    assert len(df) == 0


def test_youtube_comment_rows_parses_items():
    items = [
        {
            "snippet": {
                "channelTitle": "Ch",
                "topLevelComment": {
                    "snippet": {
                        "textOriginal": "សួស្តី",
                        "authorDisplayName": "Bunheng",
                        "publishedAt": "2026-01-01T00:00:00Z",
                        "likeCount": 3,
                    }
                },
            }
        },
        {"snippet": {"topLevelComment": {"snippet": {"textOriginal": "  "}}}},
        {},
    ]
    rows = youtube_comment_rows(items, video_id="vid1", video_title="Title")
    assert len(rows) == 1
    row = rows[0]
    assert row["text"] == "សួស្តី"
    assert row["source"] == "youtube:vid1"
    assert row["video_id"] == "vid1"
    assert row["video_title"] == "Title"
    assert row["channel"] == "Ch"
    assert row["author"] == "Bunheng"
    assert row["like_count"] == 3


def test_youtube_comment_rows_empty():
    assert youtube_comment_rows([]) == []
    assert youtube_comment_rows(None) == []


def test_maps_review_rows_parses_reviews():
    place = {"place_id": "p1", "name": "Cafe Phnom Penh"}
    reviews = [
        {"text": "តម្លៃថោក អាហារឆ្ងាញ់", "rating": 5, "author_name": "A", "time": 1787129063},
        {"text": " ", "rating": 2},
    ]
    rows = maps_review_rows(place, reviews)
    assert len(rows) == 1
    row = rows[0]
    assert row["text"] == "តម្លៃថោក អាហារឆ្ងាញ់"
    assert row["source"] == "maps:p1"
    assert row["place_name"] == "Cafe Phnom Penh"
    assert row["place_rating"] == 5


def test_maps_review_rows_empty():
    assert maps_review_rows({"place_id": "p1"}, []) == []
    assert maps_review_rows({"place_id": "p1"}, None) == []


def test_extract_video_id_various_urls():
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/live/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s") == "dQw4w9WgXcQ"
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_invalid():
    assert extract_video_id("") is None
    assert extract_video_id(None) is None
    assert extract_video_id("https://example.com/not-youtube") is None
    assert extract_video_id("https://www.youtube.com/watch?v=tooshort") is None


def test_collect_budget():
    budget = CollectBudget(10)
    assert budget.remaining() == 10
    budget.spend(4)
    budget.spend(6)
    assert budget.remaining() == 0
    try:
        budget.spend(1)
        raise AssertionError("should have raised")
    except QuotaExceededError:
        pass