"""Phase 2 — Fine-tune xlm-roberta-base as a 3-class Khmer/English/mixed sentiment model.

Train on data/splits/train.csv (neutral oversampled ~2.5x, class-weighted loss),
early-stop on val macro-F1, save best checkpoint to models/khmer-sentiment-3class/.
Test set is NOT touched here.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.common.config import SPLITS_DIR, ENCODING, SEED

MODEL_NAME = "xlm-roberta-base"
OUT_DIR = PROJECT_ROOT / "models" / "khmer-sentiment-3class"
REPORTS_DIR = PROJECT_ROOT / "reports"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}
NEUTRAL_OVERSAMPLE = 2.5


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


def oversample_neutral(df, seed=SEED):
    rng = np.random.default_rng(seed)
    neutral = df[df["label"] == "neutral"]
    target = int(len(neutral) * NEUTRAL_OVERSAMPLE)
    extra = neutral.sample(n=target - len(neutral), replace=True, random_state=seed)
    return pd.concat([df, extra], ignore_index=True)


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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "eval_accuracy": float(accuracy_score(labels, preds)),
        "eval_macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def main():
    train = pd.read_csv(SPLITS_DIR / "train.csv", encoding=ENCODING)
    val = pd.read_csv(SPLITS_DIR / "val.csv", encoding=ENCODING)

    train = train[train["label"].isin(LABELS)].copy()
    val = val[val["label"].isin(LABELS)].copy()
    train["label_id"] = train["label"].map(LABEL_TO_ID)
    val["label_id"] = val["label"].map(LABEL_TO_ID)

    counts = train["label"].value_counts().to_dict()
    counts["neutral"] = counts.get("neutral", 0)
    total = len(train)
    class_weights = [total / (len(LABELS) * counts[lab]) for lab in LABELS]
    print("class_weights:", dict(zip(LABELS, [round(w, 3) for w in class_weights])))

    train = oversample_neutral(train)
    print(f"train rows after oversample: {len(train)} "
          f"(neutral {NEUTRAL_OVERSAMPLE}x, val untouched: {len(val)})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label={i: l for i, l in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
    )

    def tokenize(texts):
        return tokenizer(
            texts.tolist(), truncation=True, max_length=MAX_LEN, padding=False
        )

    train_enc = tokenize(train["text"])
    val_enc = tokenize(val["text"])

    train_ds = TextDataset(train_enc, train["label_id"].tolist())
    val_ds = TextDataset(val_enc, val["label_id"].tolist())

    args = TrainingArguments(
        output_dir=str(REPORTS_DIR / "trainer_out"),
        seed=SEED,
        data_seed=SEED,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_steps=350,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to=[],
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    metrics = trainer.evaluate(eval_dataset=val_ds)
    print("final val metrics:", json.dumps(metrics, indent=2))

    preds = trainer.predict(val_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = val["label_id"].tolist()

    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    by_lang = {}
    for lang, mask in val.groupby("language").groups.items():
        idx = sorted(mask)
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        by_lang[lang] = {
            "rows": len(idx),
            "accuracy": round(float(accuracy_score(yt, yp)), 4),
            "macro_f1": round(float(f1_score(yt, yp, average="macro")), 4),
        }

    report = {
        "model": MODEL_NAME,
        "out_dir": str(OUT_DIR),
        "train_rows_after_oversample": len(train),
        "val_rows": len(val),
        "epochs": EPOCHS,
        "max_length": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "val": {k: round(float(v), 4) for k, v in metrics.items()},
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
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "phase2_val.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("saved model:", OUT_DIR)
    print("saved report:", REPORTS_DIR / "phase2_val.json")


if __name__ == "__main__":
    main()