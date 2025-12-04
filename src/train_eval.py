# src/train_eval.py

from typing import Dict, Any
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

def train_model(model, tokenizer, train_ds: Dataset, val_ds: Dataset, output_dir: str, num_epochs: int = 3) -> Dict[str, Any]:
    """
    Generic training function for a classification model.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    train_tokenized = train_ds.map(preprocess_function, batched=True)
    val_tokenized = val_ds.map(preprocess_function, batched=True)

    train_tokenized = train_tokenized.remove_columns(
        [c for c in train_tokenized.column_names if c not in ["input_ids", "attention_mask", "label"]]
    )
    val_tokenized = val_tokenized.remove_columns(
        [c for c in val_tokenized.column_names if c not in ["input_ids", "attention_mask", "label"]]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics
