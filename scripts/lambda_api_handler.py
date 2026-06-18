import base64
import json
import os
import re
import time
import uuid
from pathlib import PurePosixPath

import boto3
from botocore.exceptions import ClientError

runtime = boto3.client("sagemaker-runtime")
bedrock = boto3.client("bedrock-runtime")
s3 = boto3.client("s3")
sm = boto3.client("sagemaker")


DEFAULT_LABELS = [
    "aluminium",
    "batteries",
    "cardboard",
    "disposable plates",
    "glass",
    "hard plastic",
    "paper",
    "paper towel",
    "polystyrene",
    "soft plastics",
    "takeaway cups",
]


def _allowed_labels():
    """Return the label list that Bedrock is allowed to use.
    This keeps automatic labeling constrained to the same classes the model was trained on.
    """
    raw = os.environ.get("ALLOWED_LABELS_JSON")
    return json.loads(raw) if raw else DEFAULT_LABELS


def _threshold(value):
    """Convert the confidence threshold into probability format.
    This lets students pass either 0.65 or 65 and still get the same cutoff."""
    value = float(value)
    return value / 100.0 if value > 1 else value


def _headers(event):
    """Normalize API Gateway headers to lowercase keys.
    This avoids bugs caused by Content-Type casing differences between clients."""
    return {k.lower(): v for k, v in (event.get("headers") or {}).items()}


def _image_extension(content_type):
    """Choose the file extension used when saving the uploaded image to S3.
    This keeps saved examples easy to inspect later in the S3 console."""
    content_type = (content_type or "").lower()
    if "png" in content_type:
        return "png"
    if "webp" in content_type:
        return "webp"
    return "jpg"


def _bedrock_image_format(content_type):
    """Convert the incoming image content type to the format name Bedrock expects.
    This lets the same Lambda handle JPEG, PNG, and WebP uploads."""
    content_type = (content_type or "").lower()
    if "png" in content_type:
        return "png"
    if "webp" in content_type:
        return "webp"
    return "jpeg"


def _parse_request(event):
    """Extract the image bytes and confidence threshold from the API Gateway event.
    This supports both JSON base64 uploads and binary image uploads."""
    headers = _headers(event)
    content_type = headers.get("content-type", "application/json")
    query = event.get("queryStringParameters") or {}
    default_threshold = os.environ.get("CONFIDENCE_THRESHOLD", "0.65")
    threshold = _threshold(query.get("threshold", default_threshold))
    body = event.get("body") or ""

    if event.get("isBase64Encoded"):
        image_bytes = base64.b64decode(body)
        return image_bytes, content_type, threshold

    if content_type.split(";")[0].strip().lower() == "application/json":
        payload = json.loads(body)
        threshold = _threshold(payload.get("threshold", threshold))
        image_bytes = base64.b64decode(payload["image"])
        content_type = payload.get("content_type", content_type)
        return image_bytes, content_type, threshold

    return body.encode("utf-8"), content_type, threshold


def _invoke_endpoint(image_bytes):
    """Send the image to the deployed SageMaker endpoint and return its prediction.
    This is the bridge between the public API and the real-time model."""
    endpoint_name = os.environ["SAGEMAKER_ENDPOINT_NAME"]
    payload = {
        "image": base64.b64encode(image_bytes).decode("utf-8"),
        "top_k": 5,
    }
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


def _emit_metric(confidence, low_confidence, retraining_started):
    """Write model confidence and retraining signals to CloudWatch metrics.
    The dashboard reads these metrics to show whether the API is seeing weak predictions.
    """
    namespace = os.environ.get("METRIC_NAMESPACE", "RecyclingClassifierCourse")
    print(
        json.dumps(
            {
                "_aws": {
                    "Timestamp": int(time.time() * 1000),
                    "CloudWatchMetrics": [
                        {
                            "Namespace": namespace,
                            "Dimensions": [["EndpointName"]],
                            "Metrics": [
                                {"Name": "Confidence", "Unit": "None"},
                                {"Name": "LowConfidenceCount", "Unit": "Count"},
                                {"Name": "RetrainingStarted", "Unit": "Count"},
                            ],
                        }
                    ],
                },
                "EndpointName": os.environ["SAGEMAKER_ENDPOINT_NAME"],
                "Confidence": confidence,
                "LowConfidenceCount": 1 if low_confidence else 0,
                "RetrainingStarted": 1 if retraining_started else 0,
            }
        )
    )


def _put_bytes(bucket, key, data, content_type):
    """Save one byte payload to S3 with the right content type.
    This helper keeps image, prediction, and label writes consistent."""
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def _save_low_confidence_example(image_bytes, content_type, prediction, threshold):
    """Save a low-confidence API request as a pending feedback example in S3.
    This creates the training signal that Bedrock can label and the pipeline can later retrain on.
    """
    bucket = os.environ["LOW_CONFIDENCE_BUCKET"]
    prefix = os.environ.get(
        "LOW_CONFIDENCE_PREFIX", "recycling-classifier/low-confidence"
    ).strip("/")
    example_id = str(uuid.uuid4())
    base_key = f"{prefix}/pending/{example_id}"
    ext = _image_extension(content_type)

    _put_bytes(bucket, f"{base_key}/image.{ext}", image_bytes, content_type)

    metadata = {
        "example_id": example_id,
        "confidence": prediction.get("confidence"),
        "threshold": threshold,
        "saved_at_epoch": int(time.time()),
        "endpoint_name": os.environ["SAGEMAKER_ENDPOINT_NAME"],
        "prediction": prediction,
        "label_status": "pending_bedrock_label",
        "allowed_labels": _allowed_labels(),
    }
    _put_bytes(
        bucket,
        f"{base_key}/prediction.json",
        json.dumps(metadata, indent=2).encode("utf-8"),
        "application/json",
    )
    return f"s3://{bucket}/{base_key}/"


def _list_pending_examples(bucket, root_prefix):
    """Find complete pending examples that are ready for Bedrock labeling.
    A complete example has an image and prediction metadata, and no previous label error.
    """
    pending_prefix = f"{root_prefix.strip('/')}/pending/"
    paginator = s3.get_paginator("list_objects_v2")
    groups = {}
    for page in paginator.paginate(Bucket=bucket, Prefix=pending_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(pending_prefix) :]
            parts = rel.split("/", 1)
            if len(parts) != 2:
                continue
            groups.setdefault(parts[0], []).append(key)

    examples = []
    for example_id, keys in groups.items():
        names = {PurePosixPath(key).name for key in keys}
        has_image = any(name.startswith("image.") for name in names)
        has_prediction = "prediction.json" in names
        has_error = "label_error.json" in names
        if has_image and has_prediction and not has_error:
            examples.append((example_id, keys))
    return sorted(examples)


def _try_create_lock(bucket, key):
    """Create a lightweight S3 lock so only one Lambda starts retraining at a time.
    This prevents duplicate Bedrock labeling and duplicate pipeline executions during traffic bursts.
    """
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=b"locked", IfNoneMatch="*")
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"PreconditionFailed", "ConditionalRequestConflict"}:
            return False
        raise


def _delete_object(bucket, key):
    """Delete one object from S3.
    This is used to clean up locks and move pending examples after batching."""
    s3.delete_object(Bucket=bucket, Key=key)


def _read_object_bytes(bucket, key):
    """Read an S3 object and return its bytes plus content type.
    Bedrock needs the raw image bytes, while the pipeline needs the saved metadata."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    content_type = obj.get("ContentType", "image/jpeg")
    return obj["Body"].read(), content_type


def _extract_bedrock_text(response):
    """Extract plain text from the Bedrock Converse response.
    Claude returns content blocks, so this turns them into one string we can parse."""
    content = response["output"]["message"]["content"]
    return "".join(block.get("text", "") for block in content if "text" in block)


def _parse_label_json(text):
    """Parse the label JSON returned by Bedrock.
    This tolerates small formatting mistakes by extracting the first JSON object if extra text appears.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _bedrock_label_image(image_bytes, content_type, prediction):
    """Ask Bedrock to choose one valid recycling label for a low-confidence image.
    The prompt forces Claude to pick from the model's allowed labels so retraining stays supervised.
    """
    labels = _allowed_labels()
    model_id = os.environ.get("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6")
    prompt = (
        "You are labeling a recycling image for supervised image classification. "
        "Choose exactly one label from this allowed list and do not invent labels: "
        f"{labels}. "
        "Use the image as the primary evidence. The current model prediction is provided only as a hint: "
        f"{json.dumps(prediction.get('top_predictions', [])[:5])}. "
        "Return only JSON with keys label, confidence, and rationale. "
        "The label value must exactly match one allowed label."
    )
    response = bedrock.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": _bedrock_image_format(content_type),
                            "source": {"bytes": image_bytes},
                        }
                    },
                    {"text": prompt},
                ],
            }
        ],
        inferenceConfig={"maxTokens": 256, "temperature": 0},
    )
    text = _extract_bedrock_text(response)
    payload = _parse_label_json(text)
    label = str(payload.get("label", "")).strip()
    if label not in labels:
        raise ValueError(
            f"Bedrock returned invalid label {label!r}; allowed labels are {labels}"
        )
    return {
        "label": label,
        "confidence": float(payload.get("confidence", 0.0)),
        "rationale": str(payload.get("rationale", ""))[:500],
        "source": "bedrock",
        "bedrock_model_id": model_id,
        "labeled_at_epoch": int(time.time()),
    }


def _copy_selected_to_batch(
    bucket, root_prefix, batch_id, example_id, keys, label_payload
):
    """Move one labeled pending example into a retraining batch.
    This creates the stable S3 layout that train.py reads during the pipeline training step.
    """
    for key in keys:
        destination = (
            f"{root_prefix}/batches/{batch_id}/{example_id}/{PurePosixPath(key).name}"
        )
        s3.copy_object(
            Bucket=bucket, CopySource={"Bucket": bucket, "Key": key}, Key=destination
        )
        _delete_object(bucket, key)
    label_key = f"{root_prefix}/batches/{batch_id}/{example_id}/label.json"
    _put_bytes(
        bucket,
        label_key,
        json.dumps(label_payload, indent=2).encode("utf-8"),
        "application/json",
    )


def _write_pending_label(bucket, root_prefix, example_id, label_payload):
    """Save a valid Bedrock label back into the pending folder.
    This preserves good labels when a full batch cannot be formed yet."""
    label_key = f"{root_prefix}/pending/{example_id}/label.json"
    _put_bytes(
        bucket,
        label_key,
        json.dumps(label_payload, indent=2).encode("utf-8"),
        "application/json",
    )


def _write_label_error(bucket, root_prefix, example_id, error_message):
    """Record that Bedrock labeling failed for one example.
    This prevents the same bad example from being retried automatically forever."""
    key = f"{root_prefix}/pending/{example_id}/label_error.json"
    payload = {"error": error_message, "failed_at_epoch": int(time.time())}
    _put_bytes(
        bucket, key, json.dumps(payload, indent=2).encode("utf-8"), "application/json"
    )


def _maybe_label_and_start_retraining():
    """Check whether enough low-confidence examples exist, label them, and start retraining.
    This function is the automation gate between API feedback collection and the SageMaker Pipeline.
    """
    pipeline_name = os.environ.get("PIPELINE_NAME")
    if not pipeline_name:
        return {"started": False, "reason": "PIPELINE_NAME is not configured"}

    bucket = os.environ["LOW_CONFIDENCE_BUCKET"]
    root_prefix = os.environ.get(
        "LOW_CONFIDENCE_PREFIX", "recycling-classifier/low-confidence"
    ).strip("/")
    min_items = int(os.environ.get("MIN_RETRAIN_ITEMS", "3"))
    examples = _list_pending_examples(bucket, root_prefix)
    if len(examples) < min_items:
        return {
            "started": False,
            "pending_examples": len(examples),
            "required_examples": min_items,
        }

    lock_key = f"{root_prefix}/locks/retrain.lock"
    if not _try_create_lock(bucket, lock_key):
        return {"started": False, "reason": "another retraining trigger is active"}

    try:
        batch_id = str(uuid.uuid4())
        selected = examples[:min_items]
        labeled_items = []

        for example_id, keys in selected:
            names_to_keys = {PurePosixPath(key).name: key for key in keys}
            image_key = next(
                key for name, key in names_to_keys.items() if name.startswith("image.")
            )
            prediction_key = names_to_keys["prediction.json"]
            prediction = json.loads(
                _read_object_bytes(bucket, prediction_key)[0].decode("utf-8")
            )["prediction"]

            if "label.json" in names_to_keys:
                label_payload = json.loads(
                    _read_object_bytes(bucket, names_to_keys["label.json"])[0].decode(
                        "utf-8"
                    )
                )
                if label_payload.get("label") not in _allowed_labels():
                    _write_label_error(
                        bucket,
                        root_prefix,
                        example_id,
                        "stored label is not in allowed labels",
                    )
                    continue
            else:
                image_bytes, content_type = _read_object_bytes(bucket, image_key)
                try:
                    label_payload = _bedrock_label_image(
                        image_bytes, content_type, prediction
                    )
                except Exception as exc:
                    _write_label_error(bucket, root_prefix, example_id, str(exc))
                    continue

            labeled_items.append((example_id, keys, label_payload))

        if len(labeled_items) < min_items:
            for example_id, _, label_payload in labeled_items:
                _write_pending_label(bucket, root_prefix, example_id, label_payload)
            return {
                "started": False,
                "reason": "not enough valid Bedrock labels",
                "labeled_examples": len(labeled_items),
            }

        for example_id, keys, label_payload in labeled_items:
            _copy_selected_to_batch(
                bucket, root_prefix, batch_id, example_id, keys, label_payload
            )

        batch_s3_uri = f"s3://{bucket}/{root_prefix}/batches/{batch_id}/"
        parameters = [
            {"Name": "LowConfidenceS3Uri", "Value": batch_s3_uri},
        ]
        execution = sm.start_pipeline_execution(
            PipelineName=pipeline_name,
            PipelineParameters=parameters,
            PipelineExecutionDisplayName=f"bedrock-feedback-{batch_id[:8]}",
        )
        return {
            "started": True,
            "pipeline_execution_arn": execution["PipelineExecutionArn"],
            "batch_s3_uri": batch_s3_uri,
            "examples": len(labeled_items),
        }
    finally:
        _delete_object(bucket, lock_key)


def lambda_handler(event, context):
    """Handle one API Gateway request from image upload through optional retraining trigger.
    It returns the model prediction immediately and starts feedback collection when confidence is low.
    """
    image_bytes, content_type, threshold = _parse_request(event)
    prediction = _invoke_endpoint(image_bytes)
    if isinstance(prediction, list):
        prediction = prediction[0]

    if isinstance(prediction, bytes):
        prediction = prediction.decode("utf-8")

    if isinstance(prediction, str):
        prediction = json.loads(prediction)
    confidence = float(prediction["confidence"])
    low_confidence = confidence < threshold

    saved_s3_uri = None
    retraining = {"started": False}
    if low_confidence:
        saved_s3_uri = _save_low_confidence_example(
            image_bytes, content_type, prediction, threshold
        )
        retraining = _maybe_label_and_start_retraining()

    _emit_metric(confidence, low_confidence, retraining.get("started", False))

    response = {
        "predicted_label": prediction.get("predicted_label"),
        "confidence": confidence,
        "top_predictions": prediction.get("top_predictions"),
        "low_confidence": low_confidence,
        "threshold": threshold,
        "saved_s3_uri": saved_s3_uri,
        "retraining": retraining,
        "height": prediction.get("height"),
        "width": prediction.get("width"),
    }

    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(response),
    }
