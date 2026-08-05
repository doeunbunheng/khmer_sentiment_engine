"""Pipeline: language detect → (translate EN→KM) → HF sentiment → 3-class."""

from src.common.config import MODEL
from src.models.hf_api import classify_khmer
from src.models.translate_baseline import translate_en_to_km
from src.preprocessing.language_detect import detect_language
from src.common import db as common_db

THRESHOLD = float(MODEL["neutral_threshold"])


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
    translated = None
    model_input = text
    if lang == "english":
        translated = translate_en_to_km(text)
        model_input = translated
    try:
        label, score = classify_khmer(model_input)
    except Exception:
        label, score = "Negative", 0.0
    label = label.lower().strip()
    if label not in ("positive", "negative"):
        label = "negative"
    sentiment = label if score >= THRESHOLD else "neutral"
    return {
        "text": text,
        "language": lang,
        "sentiment": sentiment,
        "confidence": score,
        "translated_text": translated,
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
