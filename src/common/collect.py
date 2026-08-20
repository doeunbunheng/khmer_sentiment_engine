"""Shared helpers for data-collection scripts (YouTube comments, Google Maps reviews)."""

from pathlib import Path


def write_collected_csv(rows, path, text_column="text"):
    """Write list-of-dict rows to a UTF-8 CSV, deduplicated on the text column.

    Returns the number of rows written. The text column is always first so the
    dashboard's unseen-eval normalizer picks it up on upload.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[text_column])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
        return 0

    df = df[df[text_column].notna() & df[text_column].astype(str).str.strip().ne("")]
    df = df.drop_duplicates(subset=text_column).reset_index(drop=True)

    cols = [text_column] + [c for c in df.columns if c != text_column]
    df = df[cols]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return len(df)


def youtube_comment_rows(items, video_id="", video_title="", channel=""):
    """Convert commentThreads.list JSON items into collector CSV rows."""
    rows = []
    for item in items or []:
        snippet = item.get("snippet", {})
        top = snippet.get("topLevelComment", {}).get("snippet", {})
        text = (top.get("textOriginal") or top.get("textDisplay") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "text": text,
                "source": f"youtube:{video_id}" if video_id else "youtube",
                "video_id": video_id,
                "video_title": video_title,
                "channel": snippet.get("channelTitle") or channel,
                "author": top.get("authorDisplayName", ""),
                "published_at": top.get("publishedAt", ""),
                "like_count": top.get("likeCount", 0),
            }
        )
    return rows


def maps_review_rows(place, reviews=None):
    """Convert a Place Details object (legacy Places API) into collector CSV rows."""
    rows = []
    for review in reviews or []:
        text = (review.get("text") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "text": text,
                "source": f"maps:{place.get('place_id', '')}" if place.get("place_id") else "maps",
                "place_id": place.get("place_id", ""),
                "place_name": place.get("name", ""),
                "place_rating": review.get("rating", ""),
                "author_name": review.get("author_name", ""),
                "published_at": review.get("time", ""),
            }
        )
    return rows