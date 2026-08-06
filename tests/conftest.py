"""Global test fixtures.

Autouse: keep the ~1.2GB songhieng aspect model OUT of the test suite —
`predict_emotions` would otherwise download it on the first prediction call.
We block the download at the transformers layer, so the real `_load` logic
(label-count check, cache path) still runs and is covered by tests.
"""

import os

import pytest

os.environ.setdefault("API_SECRET", "test-secret-not-for-production")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:3000")


@pytest.fixture(autouse=True)
def _no_aspect_model_download(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("aspect model must not load in tests")

    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained", boom
    )
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", boom)
