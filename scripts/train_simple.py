# The original train script without all the retrain items in it
import json
import logging
import os
import shutil
from pathlib import Path
 
import numpy as np
import torch
from datasets import load_dataset
from datasets import load_from_disk
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification
from transformers import Trainer
from transformers import TrainingArguments
 
 
LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_ID = "google/vit-base-patch16-224-in21k"
DATASET_ID = "viola77data/recycling-dataset"
TRAIN_SPLIT = "train"
EVAL_SPLIT = ""
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"
TEST_SIZE = 0.2
EPOCHS = 2.0
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
MAX_TRAIN_SAMPLES = 2048
MAX_EVAL_SAMPLES = 128
FREEZE_BACKBONE = False
SEED = 42
DATASET_CHANNEL = "/opt/ml/input/data/dataset"


def load_base_dataset():
    dataset_channel = Path(DATASET_CHANNEL)
    if dataset_channel.exists():
        LOG.info(f"Loading staged dataset from {dataset_channel}")
        return load_from_disk(str(dataset_channel))
    LOG.info("Dataset input channel missing, loading dataset from HF")
    return load_dataset(DATASET_ID)

def finite_float(value, default=0.0):
    value = float(value)
    return value if np.isfinite(value) else default


def macro_f1_score(predictions,labels,num_labels):
    scores = []
    for label_id in range(num_labels):
        true_positive = np.sum((predictions == label_id) & (labels == label_id))
        false_positive = np.sum((predictions == label_id) & (labels != label_id))
        false_negative = np.sum((predictions != label_id) & (labels == label_id))
        support = np.sum(labels == label_id)
        if support == 0:
            continue
        precision = true_positive / max(true_positive + false_positive,1)
        recall = true_positive / max(true_positive + false_negative,1)
        score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision+recall)
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0

   
def main():
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    output_dir = Path(os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output")) / "trainer"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading dataset %s", DATASET_ID)
    raw = load_base_dataset()
    if TRAIN_SPLIT not in raw:
        raise ValueError(f"Dataset {DATASET_ID} does not contain split {TRAIN_SPLIT!r}")

    if EVAL_SPLIT and EVAL_SPLIT in raw:
        train_ds = raw[TRAIN_SPLIT]
        eval_ds = raw[EVAL_SPLIT]
    else:
        split = raw[TRAIN_SPLIT].train_test_split(test_size=TEST_SIZE, seed=SEED)
        train_ds = split["train"]
        eval_ds = split["test"]

    label_feature = train_ds.features[LABEL_COLUMN]
    label_names = list(label_feature.names)
    num_labels = len(label_names)
    id2label = {idx: name for idx, name in enumerate(label_names)}
    label2id = {name: idx for idx, name in id2label.items()}

    if MAX_TRAIN_SAMPLES:
        train_ds = train_ds.shuffle(seed=SEED).select(range(min(MAX_TRAIN_SAMPLES, len(train_ds))))
    if MAX_EVAL_SAMPLES:
        eval_ds = eval_ds.shuffle(seed=SEED).select(range(min(MAX_EVAL_SAMPLES, len(eval_ds))))

    image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    def transform_batch(batch):
        images = [image.convert("RGB") for image in batch[IMAGE_COLUMN]]
        encoded = image_processor(images=images, return_tensors="pt")
        encoded["labels"] = torch.tensor(batch[LABEL_COLUMN], dtype=torch.long)
        return encoded

    train_ds.set_transform(transform_batch)
    eval_ds.set_transform(transform_batch)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if FREEZE_BACKBONE and hasattr(model, "base_model"):
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = np.mean(predictions == labels)
        macro_f1 = macro_f1_score(predictions, labels, num_labels)
        return {"accuracy": finite_float(accuracy), "macro_f1": finite_float(macro_f1)}

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()

    metric_summary = {
        key.removeprefix("eval_"): finite_float(value)
        for key, value in eval_metrics.items()
        if key.startswith("eval_") and isinstance(value, (int, float, np.floating))
    }

    LOG.info("Evaluation metrics: %s", metric_summary)
    print(f"accuracy={metric_summary.get('accuracy', 0.0)}")

    trainer.save_model(str(model_dir))
    image_processor.save_pretrained(str(model_dir))
    with (model_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metric_summary, f, indent=2)
    with (model_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump({"labels": label_names, "id2label": id2label, "label2id": label2id}, f, indent=2)

    code_dir = model_dir / "code"
    code_dir.mkdir(exist_ok=True)
    source_dir = Path(__file__).resolve().parent
    for filename in ("inference.py", "requirements.txt"):
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, code_dir / filename)


if __name__ == "__main__":
    main()