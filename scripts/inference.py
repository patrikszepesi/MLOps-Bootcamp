import base64
import io
import json
from urllib.parse import urlparse

import boto3
import torch
from PIL import Image
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return {"model": model, "processor": processor, "device": device}


def input_fn(input_data, content_type):
    content_type = (content_type or "application/json").split(";")[0].strip().lower()
    if content_type == "application/json":
        if isinstance(input_data, bytes):
            input_data = input_data.decode("utf-8")
        return json.loads(input_data)
    if content_type in {"image/jpeg", "image/jpg", "image/png", "image/webp", "application/x-image"}:
        return {"image_bytes": input_data}
    raise ValueError(f"Unsupported content type: {content_type}")


def _load_image(payload):
    if "image_bytes" in payload:
        return Image.open(io.BytesIO(payload["image_bytes"])).convert("RGB")
    if "image" in payload:
        image_bytes = base64.b64decode(payload["image"])
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if "s3_uri" in payload:
        parsed = urlparse(payload["s3_uri"])
        if parsed.scheme != "s3":
            raise ValueError("s3_uri must start with s3://")
        obj = boto3.client("s3").get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
        return Image.open(obj["Body"]).convert("RGB")
    raise ValueError("Payload must contain one of: image, image_bytes, s3_uri")


def predict_fn(payload, bundle):
    image = _load_image(payload)
    processor = bundle["processor"]
    model = bundle["model"]
    device = bundle["device"]
    top_k = int(payload.get("top_k", 5))

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    labels = model.config.id2label
    top_count = min(top_k, probabilities.shape[0])
    top_probs, top_indices = torch.topk(probabilities, k=top_count)
    predicted_index = int(top_indices[0].item())
    predicted_label = labels[predicted_index]
    confidence = float(top_probs[0].item())

    top_predictions = [
        {"label": labels[int(index.item())], "label_id": int(index.item()), "probability": float(prob.item())}
        for prob, index in zip(top_probs, top_indices)
    ]

    return {
        "predicted_label": predicted_label,
        "predicted_label_id": predicted_index,
        "confidence": confidence,
        "top_predictions": top_predictions,
        "height": image.height,
        "width": image.width,
    }


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
