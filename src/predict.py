"""Pipeline: language detect → 3-class local sentiment → 3-class label.

Production path uses the fine-tuned `models/khmer-sentiment-3class-v2` model
(xlm-roberta) which handles Khmer / English / code-switched natively —
no EN→KM translation and no neutral-threshold rule needed.

OOD guard: every prediction carries an `uncertain` flag — `True` when the
model confidence is below `model.uncertainty_threshold` (config.yaml, default
0.90, based on calibration: rows ≥ 0.90 are well calibrated, rows below are
unreliable) or when the text is empty. The UI should treat `uncertain` as
"ask the user / show a neutral state" instead of presenting a guess.
"""

from src.common.config import UNCERTAINTY_THRESHOLD
from src.models.local_model import predict as local_predict
from src.models.aspects import predict as predict_aspects
from src.preprocessing.language_detect import detect_language
from src.common import db as common_db

LABELS = ("negative", "neutral", "positive")


def predict_sentiment(text):
    if text is None or not text.strip():
        return {
            "text": text or "",
            "language": "unknown",
            "sentiment": "neutral",
            "confidence": 0.0,
            "uncertain": True,
            "translated_text": None,
            "aspects": {"business_aspects": {}, "emotions": {"scores": {}, "active": []}},
        }
    lang = detect_language(text)
    try:
        label, score = local_predict(text)
    except Exception:
        label, score = "neutral", 0.0
    if label not in LABELS:
        label = "neutral"
    try:
        aspects = predict_aspects(text)
    except Exception:
        aspects = {"business_aspects": {}, "emotions": {"scores": {}, "active": []}}
    return {
        "text": text,
        "language": lang,
        "sentiment": label,
        "confidence": score,
        "uncertain": score < UNCERTAINTY_THRESHOLD,
        "translated_text": None,
        "aspects": aspects,
    }


def predict_and_save(text, user_id=None, consent=False, names=None):
    """Run prediction and optionally save feedback to the DB.

    Returns the prediction dict, and if saved adds `saved_id` with the
    inserted row id.
    """
    result = predict_sentiment(text)
    if user_id is None:
        return result
    # Attempt to save feedback when a user_id is provided. save_feedback
    # will validate consent and anonymization.
    try:
        saved_id = common_db.save_feedback(
            user_id=user_id,
            text_orig=text,
            lang_detect=result["language"],
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            text_translated=result.get("translated_text"),
            aspects=result.get("aspects", {}),
            consent=consent,
            names=names,
        )
        result["saved_id"] = saved_id
    except Exception:
        # Do not fail prediction on DB errors; attach an error flag.
        result["saved_error"] = True
    return result


# convenience alias
predict = predict_sentiment
