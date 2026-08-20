"""YouTube Data API v3 — reusable collector (used by scripts and the Streamlit page)."""

import os
import re

import requests
from dotenv import load_dotenv

from src.common.collect import youtube_comment_rows

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

API_KEY = os.getenv("YOUTUBE_API_KEY")
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
THREADS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

SEARCH_UNITS = 100
VIDEOS_UNITS = 1
THREADS_UNITS = 1

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class QuotaExceededError(Exception):
    pass


class CollectBudget:
    def __init__(self, budget):
        self.budget = budget
        self.used = 0

    def spend(self, units):
        if self.used + units > self.budget:
            raise QuotaExceededError(
                f"would exceed budget of {self.budget} units (used {self.used}, need {units})"
            )
        self.used += units

    def remaining(self):
        return self.budget - self.used


def extract_video_id(url_or_id):
    """Extract a YouTube video ID from a URL or a bare 11-char ID.

    Supports https://youtube.com/watch?v=ID, youtu.be/ID, shorts/ID,
    embed/ID, live/ID, or a raw ID. Returns None when unrecognized.
    """
    value = (url_or_id or "").strip()
    if not value:
        return None
    if _VIDEO_ID_RE.match(value):
        return value
    patterns = [
        r"(?:youtube\.com|youtu\.be)/(?:watch\?(?:.*&)?v=|shorts/|embed/|live/|v/)?([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return None


def get_json(session, url, params):
    params = dict(params, key=API_KEY)
    resp = session.get(url, params=params, timeout=30)
    data = resp.json()
    if resp.status_code != 200:
        errors = data.get("error", {}).get("errors", [{}])
        reason = (errors[0] or {}).get("reason", "")
        if reason == "quotaExceeded":
            raise QuotaExceededError("YouTube API daily quota (10,000 units) exhausted - try tomorrow")
        if reason == "commentsDisabled":
            return {"items": []}
        if reason == "rateLimitExceeded":
            raise QuotaExceededError("YouTube API rate limit reached - try again later")
        if reason == "forbidden" or reason == "API_KEY_SERVICE_BLOCKED":
            raise RuntimeError(
                "YouTube Data API v3 is not enabled for this key/project. "
                "Enable it at https://console.cloud.google.com/apis/library/youtube.googleapis.com "
                "and if the key is restricted, add 'YouTube Data API v3' at "
                "https://console.cloud.google.com/apis/credentials"
            )
        raise RuntimeError(f"YouTube API error {resp.status_code}: {data}")
    return data


def fetch_video_info(session, video_id, budget):
    budget.spend(VIDEOS_UNITS)
    data = get_json(session, VIDEOS_URL, {"part": "snippet", "id": video_id})
    item = (data.get("items") or [{}])[0]
    snippet = item.get("snippet", {})
    return {
        "video_id": video_id,
        "title": snippet.get("title", video_id),
        "channel": snippet.get("channelTitle", ""),
    }


def fetch_video_comments(session, video, max_comments, budget):
    params = {
        "part": "snippet",
        "videoId": video["video_id"],
        "maxResults": min(max_comments, 100),
        "order": "relevance",
        "textFormat": "plainText",
    }
    rows = []
    while len(rows) < max_comments:
        budget.spend(THREADS_UNITS)
        data = get_json(session, THREADS_URL, params)
        rows.extend(
            youtube_comment_rows(
                data.get("items", []),
                video_id=video["video_id"],
                video_title=video["title"],
                channel=video["channel"],
            )
        )
        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
    return rows[:max_comments]


def collect_for_url(url_or_id, max_comments=200, budget_units=500):
    """Collect comments for one YouTube link. Returns (video, rows, budget_used).

    Raises ValueError for a bad link, QuotaExceededError on quota issues.
    """
    video_id = extract_video_id(url_or_id)
    if not video_id:
        raise ValueError(
            "could not find a YouTube video ID in that link - paste a normal "
            "watch URL like https://www.youtube.com/watch?v=xxxxx"
        )
    if not API_KEY:
        raise ValueError(
            "YOUTUBE_API_KEY is not set in .env - add it (see .env.example)"
        )
    budget = CollectBudget(budget_units)
    session = requests.Session()
    video = fetch_video_info(session, video_id, budget)
    rows = fetch_video_comments(session, video, max_comments, budget)
    return video, rows, budget.used


def search_videos(session, query, max_videos, budget):
    params = {
        "part": "snippet",
        "type": "video",
        "q": query,
        "maxResults": min(max_videos, 50),
        "safeSearch": "none",
    }
    videos = []
    while len(videos) < max_videos:
        budget.spend(SEARCH_UNITS)
        data = get_json(session, SEARCH_URL, params)
        for item in data.get("items", []):
            video_id = item["id"].get("videoId")
            if not video_id:
                continue
            videos.append(
                {
                    "video_id": video_id,
                    "title": item["snippet"]["title"],
                    "channel": item["snippet"]["channelTitle"],
                }
            )
            if len(videos) >= max_videos:
                break
        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
    return videos