"""Phase 3 — Stacking ensemble: OOF probabilities (xlm-r 3-class + tykea 2-class)
-> LogisticRegression meta-learner.

Base model A (xlm-roberta, fine-tuned): 5-fold out-of-fold probabilities on
train.csv. Each fold trains a fresh 3-class model on 4/5 of train and predicts
the held-out 1/5 (no leakage). The same fold models predict val; probabilities
are averaged across folds.

Base model B (tykea, pretrained 2-class): never sees our labels during training,
so OOF == direct batched inference on all rows. Its (label, score) output is
mapped to a continuous 3-dim pseudo-probability (smooth version of the
score < 0.60 -> neutral rule):
    pos_p, neg_p = softmax probs of the predicted class pair
    neu_p        = 1 - max(pos_p, neg_p)

Meta features (10): [xlmr pos, neu, neg] + [tykea pos, neu, neg] +
one-hot language [khmer, english, mixed, unknown].

Meta-learner: LogisticRegression (multinomial, balanced class weights).

Test set is NOT touched here. Output -> reports/phase3_val.json,
OOF features -> reports/phase3_oof.npz (reproducibility).
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.common.config import MODEL, SPLITS_DIR, ENCODING, SEED
from src.models.local_model import _load
from src.models.translate_baseline import translate_en_to_km
from src.preprocessing.language_detect import detect_language

MODEL_NAME = "xlm-roberta-base"
XLM_DIR = PROJECT_ROOT / "models" / "khmer-sentiment-3class"
REPORTS_DIR = PROJECT_ROOT / "reports"
LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}
LANGS = ["khmer", "english", "mixed", "unknown"]
MAX_LEN = 256
BATCH_SIZE = 16
LR = 2e-5
FOLD_EPOCHS = 2
NUM_FOLDS = 5
THRESHOLD = float(MODEL["neutral_threshold"])


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def tokenize(tokenizer, texts):
    return tokenizer(
        texts.tolist(), truncation=True, max_length=MAX_LEN, padding=False
    )


def train_fold_model(train_df, fold_dir):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3,
        id2label={i: l for i, l in enumerate(LABELS)}, label2id=LABEL_TO_ID,
    )
    counts = train_df["label"].value_counts().to_dict()
    counts["neutral"] = counts.get("neutral", 0)
    total = len(train_df)
    class_weights = [total / (len(LABELS) * counts[lab]) for lab in LABELS]

    train_ds = TextDataset(tokenize(tokenizer, train_df["text"]), train_df["label_id"].tolist())

    args = TrainingArguments(
        output_dir=str(fold_dir),
        seed=SEED,
        data_seed=SEED,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=FOLD_EPOCHS,
        learning_rate=LR,
        warmup_steps=200,
        weight_decay=0.01,
        logging_steps=99999,
        save_strategy="no",
        report_to=[],
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()
    return tokenizer, model


def predict_probs(tokenizer, model, texts, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    enc = tokenize(tokenizer, pd.Series(texts))
    ds = TextDataset(enc, [0] * len(texts))
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(probs)


def tykea_to_3probs(label, score):
    """Map a tykea 2-class (label, score) pair to a 3-dim pseudo-probability
    [pos, neu, neg] that sums to 1 — smooth version of the
    score < 0.60 -> neutral rule (neutral mass grows as confidence falls)."""
    lab = str(label).lower().strip()
    sc = float(score)
    if lab.startswith("pos"):
        pos, neg = sc, 1.0 - sc
    else:
        neg, pos = sc, 1.0 - sc
    neu = 1.0 - max(pos, neg)
    total = pos + neg + neu
    return [pos / total, neu / total, neg / total]


def tykea_pseudo_probs(texts, batch_size=64):
    tokenizer, model, labels = _load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=int(MODEL["local_max_length"]), padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            for j in range(len(batch)):
                lab = labels[int(logits[j].argmax())]
                sc = float(probs[j].max())
                out.append(tykea_to_3probs(lab, sc))
    return np.array(out, dtype=np.float32)


def language_features(texts):
    feats = []
    for t in texts:
        lang = detect_language(t)
        row = [0.0] * len(LANGS)
        row[LANGS.index(lang) if lang in LANGS else LANGS.index("unknown")] = 1.0
        feats.append(row)
    return np.array(feats, dtype=np.float32)


def tykea_inputs(texts):
    inputs = []
    for t in texts:
        lang = detect_language(t)
        inp = t
        if lang == "english":
            try:
                inp = translate_en_to_km(t)
            except Exception:
                pass
        inputs.append(inp)
    return inputs


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=self.args.device)
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=NUM_FOLDS)
    ap.add_argument("--fold-epochs", type=int, default=FOLD_EPOCHS)
    ap.add_argument("--max-train-rows", type=int, default=None,
                    help="cap train rows (smoke test)")
    ap.add_argument("--keep-fold-models", action="store_true",
                    help="save fold checkpoints to models/stack_folds/")
    args = ap.parse_args()

    train = pd.read_csv(SPLITS_DIR / "train.csv", encoding=ENCODING)
    val = pd.read_csv(SPLITS_DIR / "val.csv", encoding=ENCODING)
    train = train[train["label"].isin(LABELS)].copy()
    val = val[val["label"].isin(LABELS)].copy()
    if args.max_train_rows:
        train = train.head(args.max_train_rows)
    train["label_id"] = train["label"].map(LABEL_TO_ID)
    val["label_id"] = val["label"].map(LABEL_TO_ID)
    print(f"train rows: {len(train)}  val rows: {len(val)}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    oof_xlm = np.zeros((len(train), 3), dtype=np.float32)
    val_xlm = np.zeros((len(val), 3), dtype=np.float32)

    fold_models_dir = PROJECT_ROOT / "models" / "stack_folds"
    if args.keep_fold_models:
        fold_models_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, train["label_id"]), 1):
        print(f"\n=== fold {fold}/{args.folds} ({(time.time() - t0) / 60:.1f} min elapsed) ===")
        fold_train = train.iloc[tr_idx].copy()
        fold_val = train.iloc[va_idx].copy()
        tokenizer, model = train_fold_model(fold_train, fold_models_dir / f"fold{fold}")
        if args.keep_fold_models:
            model.save_pretrained(fold_models_dir / f"fold{fold}")
            tokenizer.save_pretrained(fold_models_dir / f"fold{fold}")
        oof_xlm[va_idx] = predict_probs(tokenizer, model, fold_val["text"].tolist())
        val_xlm += predict_probs(tokenizer, model, val["text"].tolist()) / args.folds
        del model, tokenizer
        torch.cuda.empty_cache()

    print(f"\n=== base model B: tykea OOF pseudo-probs ===")
    tykea_train_inputs = tykea_inputs(train["text"].tolist())
    oof_tykea = tykea_pseudo_probs(tykea_train_inputs)
    val_tykea = tykea_pseudo_probs(tykea_inputs(val["text"].tolist()))

    lang_train = language_features(train["text"].tolist())
    lang_val = language_features(val["text"].tolist())

    X_train = np.hstack([oof_xlm, oof_tykea, lang_train])
    X_val = np.hstack([val_xlm, val_tykea, lang_val])
    y_train = train["label_id"].tolist()
    y_val = val["label_id"].tolist()

    np.savez(
        REPORTS_DIR / "phase3_oof.npz",
        oof_xlm=oof_xlm, oof_tykea=oof_tykea, lang_train=lang_train,
        val_xlm=val_xlm, val_tykea=val_tykea, lang_val=lang_val,
        y_train=np.array(y_train), y_val=np.array(y_val),
        features=["xlm_pos", "xlm_neu", "xlm_neg", "tykea_pos", "tykea_neu", "tykea_neg"] + LANGS,
    )

    print("\n=== meta-learner: LogisticRegression ===")
    meta = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED
    )
    meta.fit(X_train, y_train)
    y_pred = meta.predict(X_val)
    y_pred_train = meta.predict(X_train)

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro", labels=[0, 1, 2])
    per_class = precision_recall_fscore_support(
        y_val, y_pred, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])

    by_lang = {}
    for lang, mask in val.groupby("language").groups.items():
        idx = sorted(mask)
        yt = [y_val[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        by_lang[lang] = {
            "rows": len(idx),
            "accuracy": round(float(accuracy_score(yt, yp)), 4),
            "macro_f1": round(float(f1_score(yt, yp, average="macro")), 4),
        }

    report = {
        "model": "stacking: xlm-roberta 3class OOF + tykea 2class pseudo-probs -> LogisticRegression",
        "folds": args.folds,
        "fold_epochs": args.fold_epochs,
        "train_rows": len(train),
        "val_rows": len(val),
        "feature_names": list(np.load(REPORTS_DIR / "phase3_oof.npz")["features"]),
        "meta_learner": "LogisticRegression(C=1.0, class_weight=balanced)",
        "val": {
            "eval_accuracy": round(acc, 4),
            "eval_macro_f1": round(macro_f1, 4),
            "train_accuracy": round(float(accuracy_score(y_train, y_pred_train)), 4),
        },
        "per_class": {
            lab: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }
            for lab, p, r, f, s in zip(LABELS, *per_class)
        },
        "confusion_matrix": {
            "rows": LABELS, "cols": LABELS, "values": cm.tolist()
        },
        "by_language": by_lang,
        "elapsed_minutes": round((time.time() - t0) / 60, 1),
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "phase3_val.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\naccuracy={acc:.4f}  macro_f1={macro_f1:.4f}  "
          f"train_acc={report['val']['train_accuracy']:.4f}")
    print("per-class:")
    for lab, d in report["per_class"].items():
        print(f"  {lab:>8}: prec={d['precision']:.4f}  rec={d['recall']:.4f}  f1={d['f1']:.4f}  n={d['support']}")
    print("confusion matrix (rows=true, cols=pred):")
    print("           " + "  ".join(f"{c:>9}" for c in LABELS))
    for r, row in zip(LABELS, cm):
        print(f"{r:>9}   " + "  ".join(f"{v:>9}" for v in row))
    print("\nby language:")
    for lang, d in by_lang.items():
        print(f"  {lang:>12}: rows={d['rows']:>4}  acc={d['accuracy']:.4f}  macro_f1={d['macro_f1']:.4f}")
    print(f"\nPhase 2 comparison: acc 0.8503 / macro-F1 0.8430")
    print(f"saved: {REPORTS_DIR / 'phase3_val.json'}  ({report['elapsed_minutes']} min)")


if __name__ == "__main__":
    main()
