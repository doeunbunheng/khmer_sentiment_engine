"""AI agent tests - offline explainer, no network, no Streamlit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.ai_agent import _detect_local_llm, chat, explain

PRED = {
    "text": "The delivery was late and the price was too high",
    "language": "english",
    "sentiment": "negative",
    "confidence": 0.71,
    "uncertain": True,
    "aspects": {
        "business_aspects": {
            "Delivery": {"hit": True, "keywords": ["late", "delivery"]},
            "Price": {"hit": True, "keywords": ["price"]},
        },
        "emotions": {"scores": {"Anger": 0.8}, "active": ["Anger"]},
    },
}

DATASET = {
    "type": "dataset",
    "rows": 989,
    "errors": 0,
    "accuracy": 0.8241,
    "macro_f1": 0.7543,
    "per_class": {
        "negative": {"f1": 0.829},
        "neutral": {"f1": 0.570},
        "positive": {"f1": 0.864},
    },
    "uncertain_analysis": {"uncertain_rows": 0},
    "distribution": {"negative": 200, "neutral": 100, "positive": 689},
}

DATASET_RICH = {
    "type": "dataset",
    "rows": 250,
    "errors": 2,
    "accuracy": 0.7120,
    "macro_f1": 0.6801,
    "per_class": {
        "negative": {"f1": 0.74, "support": 90},
        "neutral": {"f1": 0.52, "support": 40},
        "positive": {"f1": 0.78, "support": 120},
    },
    "confusion_matrix": {
        "rows": ["negative", "neutral", "positive"],
        "cols": ["negative", "neutral", "positive"],
        "values": [
            [60, 15, 15],
            [20, 12, 8],
            [10, 15, 95],
        ],
    },
    "by_language": {
        "khmer": {"rows": 200, "accuracy": 0.72, "macro_f1": 0.68},
        "english": {"rows": 50, "accuracy": 0.66, "macro_f1": 0.60},
    },
    "uncertain_analysis": {
        "uncertain_rows": 30,
        "uncertain_accuracy": 0.40,
        "confident_rows": 220,
        "confident_accuracy": 0.76,
    },
    "sample_wrong": [
        {"row": 3, "text": "ទំនិញខូច មកដល់យឺត", "true": "negative",
         "pred": "neutral", "conf": 0.58, "uncertain": True},
        {"row": 7, "text": "okay product but expensive", "true": "neutral",
         "pred": "positive", "conf": 0.91, "uncertain": False},
    ],
    "distribution": {"negative": 85, "neutral": 42, "positive": 121},
}

PREDICTIONS = [
    {"row": 1, "text": "ផលិតផលល្អណាស់", "language": "khmer",
     "sentiment": "positive", "confidence": 0.96, "uncertain": False},
    {"row": 2, "text": "ដឹកយូរពេក", "language": "khmer",
     "sentiment": "negative", "confidence": 0.93, "uncertain": False,
     "aspects": {"business_aspects": {"Delivery": {"hit": True,
                 "keywords": ["ដឹក"]}}, "emotions": {"active": ["Anger"]}}},
    {"row": 3, "text": "អត់ដឹងថាល្អឬអត់", "language": "khmer",
     "sentiment": "neutral", "confidence": 0.61, "uncertain": True},
    {"row": 4, "text": "delivery very slow", "language": "english",
     "sentiment": "negative", "confidence": 0.79, "uncertain": True,
     "aspects": {"business_aspects": {"Delivery": {"hit": True,
                 "keywords": ["delivery"]}}, "emotions": {}}},
    {"row": 5, "text": "សេវាកម្មល្អ", "language": "khmer",
     "sentiment": "positive", "confidence": 0.98, "uncertain": False},
    {"row": 6, "text": "កម្ម៉ង់ពណ៌ខៀវ បែរជាផ្ញើពណ៌ក្រហមមកវិញ",
     "language": "khmer",
     "sentiment": "negative", "confidence": 0.97, "uncertain": False},
]

FACEBOOK_DATASET = {
    "type": "dataset",
    "dataset": "facebook_comments.csv",
    "rows": 21,
    "errors": 0,
    "accuracy": None,
    "macro_f1": None,
    "distribution": {"negative": 3, "neutral": 4, "positive": 14},
    "uncertain_rows": 5,
    "predictions": PREDICTIONS,
}


def test_no_context_explainable():
    out = explain("why", None)
    assert "don't have a result" in out


def test_why_negative():
    out = explain("why is this negative?", PRED)
    assert "negative" in out
    assert "Delivery" in out and "Price" in out
    assert "71%" in out or "0.71" in out


def test_uncertain_question():
    out = explain("is it uncertain? should I trust it?", PRED)
    assert "uncertain" in out
    assert "needs review" in out


def test_aspects_question():
    out = explain("what topics is this about?", PRED)
    assert "Delivery" in out and "Price" in out


def test_emotions_question():
    out = explain("what emotions fired?", PRED)
    assert "Anger" in out


def test_metrics_question():
    out = explain("what are the metrics?", DATASET)
    assert "82.4" in out
    assert "macro-f1" in out.lower()


def test_what_happened_single_prediction():
    out = explain("what happened on this result?", PRED)
    assert "negative" in out
    assert "Delivery" in out and "Price" in out
    assert "71%" in out or "0.71" in out


def test_why_did_it_show_single_prediction():
    out = explain("why did it show like this?", PRED)
    assert "negative" in out
    assert "Delivery" in out
    assert "no dataset-wide metrics" not in out


def test_metrics_question_single_prediction_explains_result():
    out = explain("what are the metrics?", PRED)
    assert "negative" in out
    assert "no dataset-wide metrics" not in out


def test_clear_summary_dataset():
    out = explain("give me a short summary", DATASET)
    assert "accuracy" in out.lower()


def test_chat_falls_back_to_local_without_api(monkeypatch):
    monkeypatch.delenv("AGENT_API_URL", raising=False)
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    out = chat("why positive?", PRED)
    assert isinstance(out, str)
    assert out


def test_chat_uses_llm_when_configured(monkeypatch):
    monkeypatch.setenv("AGENT_API_URL", "http://fake/v1")
    monkeypatch.setenv("AGENT_API_KEY", "k")

    def fake_post(url, headers=None, json=None, timeout=None):
        class R:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "LLM ANSWER"}}]}

        return R()

    monkeypatch.setattr("app.ai_agent.requests.post", fake_post)
    assert chat("why?", PRED) == "LLM ANSWER"


def test_dataset_generic_question_answers_benchmark():
    out = explain("what happened on this dataset?", DATASET)
    assert "82.4" in out
    assert "Most were predicted" in out


def test_dataset_without_accuracy_explains_prediction_overview():
    unlabeled = {
        "type": "dataset",
        "rows": 31,
        "accuracy": None,
        "macro_f1": None,
        "distribution": {"negative": 11, "neutral": 9, "positive": 11},
        "uncertain_rows": 0,
    }
    out = explain("what happened on this dataset?", unlabeled)
    assert "31" in out
    assert "Most were predicted" in out
    # missing labels are NOT pushed on a generic question - only on
    # accuracy/correctness questions
    assert "no true labels" not in out


def test_dataset_improve_question_advises():
    out = explain("what should I do next?", DATASET)
    assert "Based on the customer feedback" in out
    assert "signal to check, not proof" in out


def test_dataset_confusion_question_uses_matrix():
    out = explain("show me the confusion matrix", DATASET_RICH)
    assert "neutral" in out
    assert "negative" in out
    assert "confusions" in out.lower()


def test_dataset_confusion_question_no_confusions():
    clean = {
        "type": "dataset",
        "accuracy": 0.95,
        "confusion_matrix": {
            "rows": ["negative", "neutral", "positive"],
            "cols": ["negative", "neutral", "positive"],
            "values": [[300, 0, 0], [0, 200, 0], [0, 0, 400]],
        },
    }
    out = explain("show me the confusion matrix", clean)
    assert "no notable confusion" in out.lower()


def test_dataset_wrong_question_lists_examples():
    out = explain("what went wrong? show me mistakes", DATASET_RICH)
    assert "Row 3" in out
    assert "Row 7" in out
    assert "true" in out


def test_dataset_weak_question_identifies_weakest_class():
    out = explain("which classes is it weak at?", DATASET_RICH)
    assert "neutral" in out
    assert "0.52" in out


def test_dataset_language_question_breaks_down_languages():
    out = explain("how is it per language?", DATASET_RICH)
    assert "khmer" in out
    assert "english" in out
    assert "weakest" in out


def test_dataset_recommendations_are_data_driven():
    out = explain("what should I do next?", DATASET_RICH)
    assert "Based on the customer feedback" in out
    assert "To improve this" not in out  # business advice, not ML advice
    assert "signal to check, not proof" in out


def test_dataset_improve_model_question_advises_technical():
    out = explain("how can I improve the model?", DATASET_RICH)
    assert "To improve this" in out
    assert "uncertain" in out.lower()


def test_business_recommendations_use_complaint_topics():
    out = explain("what should I do next?", FACEBOOK_DATASET)
    assert "3 comments were negative" in out
    assert "Delivery" in out
    assert "wrong item or color" in out  # wrong-order check detected
    assert "retrain" not in out
    assert "true labels" not in out


def test_business_recommendations_mentions_uncertain_review():
    out = explain("what should I check?", FACEBOOK_DATASET)
    assert "not sure" in out
    assert "read those by hand" in out


def test_business_recommendations_no_negative_rows():
    only_pos = {
        "type": "dataset",
        "rows": 4,
        "distribution": {"positive": 4, "neutral": 0, "negative": 0},
        "uncertain_rows": 0,
        "predictions": [
            {"row": 1, "text": "love it", "sentiment": "positive",
             "confidence": 0.99, "uncertain": False},
        ],
    }
    out = explain("what should I do next?", only_pos)
    assert "no negative" in out
    assert "signal to check, not proof" in out


def test_dataset_happened_question_is_friendly_overview():
    out = explain("what happened on this dataset?", DATASET_RICH)
    assert "71.2" in out
    assert "Most were predicted" in out
    assert "30" in out  # uncertain rows


def test_dataset_digest_includes_weak_confusion_recommendations():
    out = explain("give me the full picture of this result", DATASET_RICH)
    assert "71.2" in out
    assert "weakest" in out or "weak spot" in out
    assert "Based on the customer feedback" in out


def test_unknown_question_gets_full_digest_prediction():
    out = explain("what else can you tell me", PRED)
    assert "negative" in out
    assert "language" in out.lower()
    assert "Delivery" in out


def test_unknown_question_gets_full_digest_dataset():
    out = explain("random chatty line", DATASET)
    assert "82.4" in out
    assert "Based on the customer feedback" in out


def test_chat_override_wins_over_env_when_failing(monkeypatch):
    monkeypatch.delenv("AGENT_API_URL", raising=False)
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    out = chat("why positive?", PRED, url=None, key=None, model=None)
    assert isinstance(out, str)
    assert "positive" in out


def test_chat_llm_overrides_from_ui(monkeypatch):
    monkeypatch.delenv("AGENT_API_URL", raising=False)
    monkeypatch.delenv("AGENT_API_KEY", raising=False)

    def fake_post(url, headers=None, json=None, timeout=None):
        assert json["model"] == "gemini-2.0-flash"
        assert "Bearer ui-key" in headers["Authorization"]

        class R:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "UI LLM"}}]}

        return R()

    monkeypatch.setattr("app.ai_agent.requests.post", fake_post)
    assert chat(
        "why?", PRED,
        url="https://example/v1", key="ui-key", model="gemini-2.0-flash",
    ) == "UI LLM"


# ---- per-row dataset questions ---------------------------------------------

def test_why_most_comments_positive():
    out = explain("why are most comments positive?", FACEBOOK_DATASET)
    assert "positive" in out
    assert "14 of 21" in out
    assert "Comment #1" in out  # real examples from the rows


def test_why_most_comments_negative():
    out = explain("why are most comments negative?", FACEBOOK_DATASET)
    # the data says most are actually positive - the agent corrects the
    # premise instead of inventing a negative majority
    assert "Most comments were predicted **positive**" in out
    assert "3 negative" in out


def test_show_uncertain_comments_lists_rows():
    out = explain("show me the uncertain comments", FACEBOOK_DATASET)
    assert "2 uncertain comments" in out
    assert "Comment #3" in out and "61%" in out
    assert "Comment #4" in out and "79%" in out
    assert "Comment #3" in out  # lowest confidence first


def test_why_do_i_have_uncertain_comments():
    out = explain("why do I have 5 uncertain comments?", FACEBOOK_DATASET)
    assert "2 comments were flagged" in out
    assert "90%" in out
    assert "Comment #3" in out


def test_why_is_comment_3_uncertain():
    out = explain("why is comment 3 uncertain?", FACEBOOK_DATASET)
    assert "Comment #3" in out
    assert "អត់ដឹងថាល្អឬអត់" in out
    assert "61%" in out
    assert "uncertain" in out


def test_what_are_customers_complaining_about():
    out = explain("what are customers complaining about?", FACEBOOK_DATASET)
    assert "negative" in out
    assert "Delivery" in out  # aspect topics mined from the negative rows
    assert "Comment #2" in out and "Comment #4" in out


def test_is_model_doing_well_without_labels():
    out = explain("is the model doing well?", FACEBOOK_DATASET)
    assert "does not contain true labels" in out
    assert "accuracy cannot" in out


def test_is_model_doing_well_with_labels():
    out = explain("is the model doing well?", DATASET)
    assert "82.4%" in out
    assert "solid" in out


def test_what_happened_on_facebook_dataset():
    out = explain("what happened in this dataset?", FACEBOOK_DATASET)
    assert "21" in out
    assert "positive" in out
    assert "5 predictions were uncertain" in out
    assert "no true labels" not in out


def test_unknown_row_number_says_not_found():
    out = explain("what about comment 99?", FACEBOOK_DATASET)
    assert "no comment #99" in out


def test_show_all_negative_comments_lists_every_row():
    out = explain("show at all about comment negative", FACEBOOK_DATASET)
    assert "There are 3 comments predicted **negative**" in out
    assert "Comment #2" in out
    assert "Comment #4" in out
    assert "Comment #6" in out


def test_class_answer_notes_more_rows_exist():
    many_neg = {
        "type": "dataset",
        "rows": 9,
        "distribution": {"negative": 5, "neutral": 2, "positive": 2},
        "predictions": [
            {"row": i, "text": "bad thing #{}".format(i),
             "sentiment": "negative", "confidence": 0.9, "uncertain": False}
            for i in range(1, 6)
        ] + [
            {"row": i, "text": "ok #{}".format(i),
             "sentiment": "neutral", "confidence": 0.9, "uncertain": False}
            for i in range(6, 8)
        ] + [
            {"row": i, "text": "great #{}".format(i),
             "sentiment": "positive", "confidence": 0.9, "uncertain": False}
            for i in range(8, 10)
        ],
    }
    out = explain("why are comments negative?", many_neg)
    assert "**5** of 9" in out
    assert "Comment #1" in out
    assert "and 2 more" in out


def test_show_all_positive_comments():
    out = explain("show me all the positive comments", FACEBOOK_DATASET)
    assert "There are 2 comments predicted **positive**" in out
    assert "Comment #1" in out and "Comment #5" in out


def test_show_negative_when_none_exist():
    only_pos = {
        "type": "dataset",
        "rows": 4,
        "distribution": {"positive": 4, "neutral": 0, "negative": 0},
        "predictions": [
            {"row": 1, "text": "love it", "sentiment": "positive",
             "confidence": 0.99, "uncertain": False},
        ],
    }
    out = explain("show me the negative comments", only_pos)
    assert "No comments were predicted **negative**" in out


NON_ECOMMERCE = {
    "type": "dataset",
    "rows": 5,
    "distribution": {"negative": 3, "neutral": 1, "positive": 1},
    "uncertain_rows": 0,
    "predictions": [
        {"row": 1, "text": "delivery is very slow, waited an hour",
         "sentiment": "negative", "confidence": 0.92, "uncertain": False},
        {"row": 2, "text": "delivery never arrived at all",
         "sentiment": "negative", "confidence": 0.91, "uncertain": False},
        {"row": 3, "text": "the food was cold and tasteless",
         "sentiment": "negative", "confidence": 0.90, "uncertain": False},
        {"row": 4, "text": "it was okay", "sentiment": "neutral",
         "confidence": 0.95, "uncertain": False},
        {"row": 5, "text": "great food", "sentiment": "positive",
         "confidence": 0.98, "uncertain": False},
    ],
}


def test_complaints_answer_discovers_topics_from_text_without_aspects():
    # dataset about a restaurant - none of the fixed business aspects hit,
    # so the agent must learn the topic from the words themselves
    out = explain("what are customers complaining about?", NON_ECOMMERCE)
    assert "words customers mention most" in out
    assert "**delivery**" in out  # repeated 2x across the negative rows
    assert "tasteless" in out  # single-use topic still visible in the examples


def test_business_recommendations_use_text_terms_when_no_aspects():
    out = explain("what should I do next?", NON_ECOMMERCE)
    assert "3 comments were negative" in out
    assert "delivery" in out
    assert "Look into what people describe" in out
    assert "retrain" not in out


def test_complaints_answer_khmer_terms():
    khmer_neg = {
        "type": "dataset",
        "rows": 3,
        "distribution": {"negative": 2, "neutral": 1, "positive": 0},
        "predictions": [
            {"row": 1, "text": "ដឹកជញ្ជូនយឺតណាស់ ចាំយូរ", "sentiment": "negative",
             "confidence": 0.9, "uncertain": False},
            {"row": 2, "text": "ដឹកជញ្ជូនមិនទាន់ដល់ទេ អស់សង្ឃឹម", "sentiment": "negative",
             "confidence": 0.9, "uncertain": False},
            {"row": 3, "text": "ផ្ទះបាយឆ្ងាញ់", "sentiment": "neutral",
             "confidence": 0.9, "uncertain": False},
        ],
    }
    out = explain("what are customers complaining about?", khmer_neg)
    assert "words customers mention most" in out
    assert "**ដឹកជញ្ជូន**" in out  # segmented Khmer term repeated in both rows


def test_terms_of_keeps_khmer_words_with_combining_marks():
    from app.ai_agent import _terms_of
    t = "ព័ត៌មាននេះក្លែងក្លាយទេ អត់មានភស្តុតាង"
    terms = _terms_of(t)
    assert "ព័ត៌មាន" in terms  # contains combining mark ៌
    assert "ក្លែងក្លាយ" in terms  # contains coeng ្
    assert "នេះ" not in terms      # stopword
    assert "អត់" not in terms       # stopword


def test_unrelated_domain_terms_are_discovered():
    # politics/news data - nothing to do with the 5 business aspects,
    # but repeated words must still surface as the topics
    rows = [
        {"row": 1, "text": "ព័ត៌មាននេះក្លែងក្លាយទេ អត់មានភស្តុតាង",
         "sentiment": "negative", "confidence": 0.9, "uncertain": False},
        {"row": 2, "text": "ព័ត៌មានក្លែងក្លាយ បោកប្រាស់ប្រជាពលរដ្ឋ",
         "sentiment": "negative", "confidence": 0.9, "uncertain": False},
        {"row": 3, "text": "អ្នកយកព័ត៌មានមិនមានក្រមសីលធម៌",
         "sentiment": "negative", "confidence": 0.9, "uncertain": False},
    ]
    result = {"type": "dataset", "rows": 3,
              "distribution": {"negative": 3}, "predictions": rows}
    out = explain("what are customers complaining about?", result)
    assert "words customers mention most" in out
    assert "**ព័ត៌មាន**" in out  # in all 3 rows
    assert "**ក្លែងក្លាយ**" in out  # in 2 rows


def test_detect_local_llm_when_ollama_running(monkeypatch):
    class Ok:
        status_code = 200

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: Ok())
    url, model = _detect_local_llm()
    assert url == "http://127.0.0.1:11434/v1"
    assert model == "qwen2.5:3b"


def test_detect_local_llm_when_not_running(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError()

    monkeypatch.setattr(requests, "get", boom)
    assert _detect_local_llm() == (None, None)


def test_class_answer_one_sentence_and_action_for_many_negatives():
    rows = []
    for i in range(7):
        rows.append({
            "row": i + 1,
            "text": "comment {} delivery too slow".format(i + 1),
            "sentiment": "negative", "confidence": 0.95, "uncertain": False,
            "aspects": {"business_aspects": {
                "Delivery": {"hit": True}, "Price": {"hit": False}}},
        })
    result = {
        "type": "dataset", "rows": 7,
        "distribution": {"negative": 7},
        "predictions": rows,
    }
    out = explain(
        "why the comment negative what should i do on the comment negative",
        result,
    )
    assert "In one sentence" in out
    assert "**Delivery**" in out
    assert "What to do" in out
    assert "comment 1 delivery too slow" in out  # short example
    assert "and 4 more" in out  # 7 - 3 examples


def test_single_comment_advice_when_asked():
    result = {
        "type": "single", "row": 8,
        "text": "ខកចិត្តខ្លាំងណាស់ កម្ម៉ង់អាវផ្ញើខោមកឱ្យខ្ញុំទៅវិញ",
        "sentiment": "negative", "confidence": 0.99, "uncertain": False,
    }
    out = explain("why this comment negative 100%? give suggest idea for me",
                  result)
    assert "The model classified this comment as negative" in out
    assert "Suggested idea" in out
    assert "verify" in out.lower()


def test_long_pasted_list_is_truncated_not_echoed():
    blob = "\n".join(
        "Comment #{} — negative — 95% \"some text {}\"".format(i, i)
        for i in range(10)
    )
    result = {
        "type": "single", "row": 0, "text": blob,
        "sentiment": "negative", "confidence": 0.95, "uncertain": False,
    }
    out = explain("why is this the result?", result)
    assert "long text" in out
    assert "419 characters" in out
    assert "Comment #9" not in out  # not the whole blob echoed


EDUCATION_ROWS = [
    {"row": 1, "text": "សិស្សខ្មែររៀនដើម្បីចង់បានត្រឹមសញ្ញាបត្រ័ មិនមែនដើម្បីចំណេះពិតប្រាកដ",
     "sentiment": "negative", "confidence": 0.93, "uncertain": False,
     "aspects": {"business_aspects": {
         "Authenticity": {"hit": True, "keywords": ["ពិត"]},
         "Product Quality": {"hit": False}, "Price": {"hit": False},
         "Service": {"hit": False}, "Delivery": {"hit": False}}}},
    {"row": 2, "text": "មុខវិជ្ជាវិទ្យាសាស្ត្រនៅខ្មែរ ខំរៀនចង់ឆ្កួតមនុស្ស ប្រាក់ខែវិញដូចអាចម៍",
     "sentiment": "negative", "confidence": 0.97, "uncertain": False,
     "aspects": {"business_aspects": {
         "Authenticity": {"hit": False}, "Product Quality": {"hit": False},
         "Price": {"hit": True, "keywords": ["តម្លៃ"]},
         "Service": {"hit": False}, "Delivery": {"hit": False}}}},
    {"row": 3, "text": "អ្នកយកព័ត៌មានមិនមានក្រមសីលធម៌",
     "sentiment": "negative", "confidence": 0.98, "uncertain": False,
     "aspects": {"business_aspects": {
         "Authenticity": {"hit": False}, "Product Quality": {"hit": False},
         "Price": {"hit": False}, "Service": {"hit": False},
         "Delivery": {"hit": False}}}},
]


def test_education_dataset_gets_no_shopping_advice():
    result = {"type": "dataset", "rows": 3,
              "distribution": {"negative": 3},
              "predictions": EDUCATION_ROWS}
    out = explain("what should i do next?", result)
    assert "shipping" not in out
    assert "packing" not in out
    assert "supplier" not in out
    assert "customer-service record" not in out
    assert "words customers mention most" in out
    assert "ខ្មែរ" in out  # the real repeated topic, not the aspects


def test_education_dataset_one_sentence_uses_terms_not_aspects():
    result = {"type": "dataset", "rows": 3,
              "distribution": {"negative": 3},
              "predictions": EDUCATION_ROWS}
    out = explain("why are comments negative?", result)
    assert "mostly about **Authenticity**" not in out
    assert "ព័ត៌មាន" in out


def test_commerce_dataset_still_gets_shopping_advice():
    rows = [
        {"row": 1, "text": "ដឹកជញ្ជូនយឺតណាស់ ចាំជិតមួយអាទិត្យហើយ",
         "sentiment": "negative", "confidence": 0.95, "uncertain": False,
         "aspects": {"business_aspects": {
             "Delivery": {"hit": True, "keywords": ["ដឹកជញ្ជូន"]}}}},
        {"row": 2, "text": "អាវបោកទៅរួញចូលគ្នា គុណភាពអន់ខ្លាំង",
         "sentiment": "negative", "confidence": 0.95, "uncertain": False,
         "aspects": {"business_aspects": {
             "Product Quality": {"hit": True, "keywords": ["គុណភាព"]}}}},
    ]
    result = {"type": "dataset", "rows": 2,
              "distribution": {"negative": 2}, "predictions": rows}
    out = explain("what should i do next?", result)
    assert "The main complaints are about" in out
    assert "**Delivery**" in out
    assert "**Product Quality**" in out
    assert "customer-service record" in out


def test_followup_why_resolves_last_mentioned_row():
    history = [{"role": "user", "content": "why is comment 3 uncertain?"}]
    out = explain("why?", FACEBOOK_DATASET, history=history)
    assert "Comment #3" in out
    assert "អត់ដឹងថាល្អឬអត់" in out


def test_followup_what_about_this_one_resolves_row():
    history = [
        {"role": "user", "content": "show me the uncertain comments"},
        {"role": "assistant", "content": "Comment #3 and Comment #4 are uncertain."},
        {"role": "user", "content": "why is comment 4 uncertain?"},
    ]
    out = explain("what about this one?", FACEBOOK_DATASET, history=history)
    assert "Comment #4" in out
    assert "delivery very slow" in out


def test_followup_does_not_steal_numbers_from_assistant_answers():
    history = [
        {"role": "assistant", "content": "Comment #3 is the lowest at 61%."},
    ]
    out = explain("why do I have 5 uncertain comments?", FACEBOOK_DATASET,
                  history=history)
    # answered dataset-wide (count from the data), not hijacked into a
    # row-3 answer by the assistant's own mention of comment #3
    assert "2 comments were flagged" in out
    assert "below the 90% bar" in out


def test_followup_without_history_falls_back_to_digest():
    out = explain("why?", FACEBOOK_DATASET)
    assert "21" in out


def test_row_answer_includes_true_label():
    labeled = {**FACEBOOK_DATASET, "accuracy": 0.8,
               "predictions": [{
                   **PREDICTIONS[1],
                   "true": "negative",
               }, *PREDICTIONS[:1], *PREDICTIONS[2:]]}
    out = explain("is comment 2 correct?", labeled)
    assert "true label" in out
    assert "correct" in out


def test_llm_receives_history(monkeypatch):
    monkeypatch.delenv("AGENT_API_URL", raising=False)
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        assert json["messages"][0]["role"] == "system"
        assert "ACTIVE DATA" in json["messages"][0]["content"]
        assert json["messages"][-1] == {"role": "user", "content": "why?"}
        hist = [m for m in json["messages"] if m["role"] == "user"]
        assert any("comment 3" in m["content"] for m in hist)

        class R:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "LLM"}}]}

        return R()

    monkeypatch.setattr("app.ai_agent.requests.post", fake_post)
    out = chat(
        "why?",
        FACEBOOK_DATASET,
        url="http://fake/v1", key="k", model="m",
        history=[{"role": "user", "content": "why is comment 3 uncertain?"}],
    )
    assert out == "LLM"
    assert captured["json"]["messages"][0]["content"].count("row 1") >= 1
    assert "distribution" in captured["json"]["messages"][0]["content"]