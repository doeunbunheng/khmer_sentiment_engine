"""Pipeline: language detect → 3-class local sentiment → 3-class label.

Production path uses the fine-tuned `models/khmer-sentiment-3class` model
(xlm-roberta) which handles Khmer / English / code-switched natively —
no EN→KM translation and no neutral-threshold rule needed.
"""

from src.models.local_model import predict as local_predict
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
            "translated_text": None,
        }
    lang = detect_language(text)
    try:
        label, score = local_predict(text)
    except Exception:
        label, score = "neutral", 0.0
    if label not in LABELS:
        label = "neutral"
    return {
        "text": text,
        "language": lang,
        "sentiment": label,
        "confidence": score,
        "translated_text": None,
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
