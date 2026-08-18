"""HTTP client for the Khmer Sentiment Engine FastAPI server.

Used by the Streamlit dashboard (app/dashboard.py). Pure `requests`-based —
no Streamlit import — so it is unit-tested with mocked HTTP in
tests/test_dashboard.py.

The token is never stored on the shared cached client; callers pass it
explicitly per request to avoid leaking one session's token to another user
via Streamlit's `st.cache_resource`.
"""

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 120


class ApiError(Exception):
    """Raised on network failure or a non-2xx API response."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class ApiClient:
    def __init__(self, base_url=DEFAULT_BASE_URL, timeout=TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ---- low level ---------------------------------------------------------

    def _request(self, method, path, json=None, bearer=None):
        url = f"{self.base_url}{path}"
        headers = {}
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        try:
            r = requests.request(
                method, url, json=json, headers=headers, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise ApiError(f"cannot reach API at {self.base_url}: {exc}") from exc
        if r.status_code >= 400:
            detail = ""
            try:
                detail = r.json().get("detail", "")
            except Exception:
                detail = ""
            message = detail or (r.text or "")[:200]
            raise ApiError(f"API error {r.status_code}: {message}", status=r.status_code)
        return r.json()

    # ---- endpoints ---------------------------------------------------------

    def health(self):
        return self._request("GET", "/health", bearer=None)

    def login(self, username, password):
        return self._request(
            "POST",
            "/auth/login",
            json={"username": username, "password": password},
            bearer=None,
        )

    def register(self, username, password, full_name=None, email=None, phone=None):
        body = {"username": username, "password": password}
        if full_name:
            body["full_name"] = full_name
        if email:
            body["email"] = email
        if phone:
            body["phone"] = phone
        return self._request("POST", "/auth/register", json=body, bearer=None)

    def predict(self, text, consent=False, token=None):
        return self._request(
            "POST",
            "/predict",
            json={"text": text, "consent": consent},
            bearer=token,
        )

    def feedback(self, limit=50, token=None):
        return self._request("GET", f"/feedback?limit={limit}", bearer=token)