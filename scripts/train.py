import json
import logging
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import torch
from datasets import ClassLabel
from datasets import Dataset as HFDataset
from datasets import Features
from datasets import Image as HFImage
from datasets import concatenate_datasets
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
EPOCHS = 1.0
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
MAX_TRAIN_SAMPLES = 512
MAX_EVAL_SAMPLES = 128
SEED = 42
FREEZE_BACKBONE = False
LOW_CONFIDENCE_S3_URI = ""
HYPERPARAMETERS_PATH = "/opt/ml/input/config/hyperparameters.json"
DATASET_CHANNEL = "/opt/ml/input/data/dataset"


def load_sagemaker_hyperparameters():
    path = Path(HYPERPARAMETERS_PATH)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_feedback_batch_uri():
    hyperparameters = load_sagemaker_hyperparameters()
    return os.environ.get("LOW_CONFIDENCE_S3_URI") or hyperparameters.get("low_confidence_s3_uri", LOW_CONFIDENCE_S3_URI)


def load_base_dataset():
    dataset_channel = Path(DATASET_CHANNEL)
    if dataset_channel.exists():
        LOG.info("Loading staged dataset from %s", dataset_channel)
        return load_from_disk(str(dataset_channel))
    LOG.info("Dataset input channel missing; loading %s from Hugging Face", DATASET_ID)
    return load_dataset(DATASET_ID)


def parse_s3_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def list_s3_keys(s3_uri):
    bucket, prefix = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield bucket, obj["Key"]


def read_json_from_s3(bucket, key):
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def download_extra_examples(s3_uri, local_dir, image_column, label_column, label_names):
    if not s3_uri:
        return None

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    groups = {}
    _, root_prefix = parse_s3_uri(s3_uri)
    label_set = set(label_names)

    for bucket, key in list_s3_keys(s3_uri):
        if key.endswith("/"):
            continue
        rel = key[len(root_prefix) :].lstrip("/")
        example_id = rel.split("/", 1)[0] if "/" in rel else Path(rel).stem
        groups.setdefault(example_id, []).append((bucket, key))

    image_paths = []
    labels = []
    label_to_id = {name: idx for idx, name in enumerate(label_names)}

    for example_id, objects in sorted(groups.items()):
        image_obj = None
        label_obj = None

        for bucket, key in objects:
            name = Path(key).name.lower()
            suffix = Path(key).suffix.lower()
            if name == "label.json":
                label_obj = (bucket, key)
            elif name.startswith("image") and suffix in {".jpg", ".jpeg", ".png", ".webp"}:
                image_obj = (bucket, key)

        if not image_obj or not label_obj:
            LOG.info("Skipping incomplete feedback example %s", example_id)
            continue

        label_payload = read_json_from_s3(label_obj[0], label_obj[1])
        label_name = str(label_payload.get("label", "")).strip()
        if label_name not in label_set:
            LOG.warning("Skipping %s with invalid label %r", example_id, label_name)
            continue

        target_dir = local_dir / example_id
        target_dir.mkdir(parents=True, exist_ok=True)
        image_path = target_dir / Path(image_obj[1]).name
        s3.download_file(image_obj[0], image_obj[1], str(image_path))
        image_paths.append(str(image_path))
        labels.append(label_to_id[label_name])

    if not image_paths:
        LOG.info("No valid Bedrock-labeled feedback examples found under %s", s3_uri)
        return None

    features = Features({image_column: HFImage(), label_column: ClassLabel(names=label_names)})
    LOG.info("Loaded %d Bedrock-labeled feedback examples from %s", len(image_paths), s3_uri)
    return HFDataset.from_dict({image_column: image_paths, label_column: labels}, features=features)


def finite_float(value, default=0.0):
    value = float(value)
    return value if np.isfinite(value) else default


def macro_f1_score(predictions, labels, num_labels):
    scores = []
    for label_id in range(num_labels):
        true_positive = np.sum((predictions == label_id) & (labels == label_id))
        false_positive = np.sum((predictions == label_id) & (labels != label_id))
        false_negative = np.sum((predictions != label_id) & (labels == label_id))
        support = np.sum(labels == label_id)
        if support == 0:
            continue
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def main():
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    output_dir = Path(os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output")) / "trainer"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    low_confidence_s3_uri = get_feedback_batch_uri()

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

    extra_ds = download_extra_examples(
        low_confidence_s3_uri,
        "/tmp/bedrock-feedback-examples",
        IMAGE_COLUMN,
        LABEL_COLUMN,
        label_names,
    )
    if extra_ds is not None:
        train_ds = concatenate_datasets([train_ds, extra_ds])

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
