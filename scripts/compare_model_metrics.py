import json
import logging
import os
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError







LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_DIR = "/opt/ml/processing/model"
OUTPUT_DIR = "/opt/ml/processing/evaluation"
METRIC_NAME = "accuracy"
MIN_DELTA = 0.0
CURRENT_METRICS_S3_URI = os.environ.get("CURRENT_METRICS_S3_URI", "")




def parse_s3_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")




def load_previous_metrics(s3_uri):
    if not s3_uri:
        return {}
    bucket, key = parse_s3_uri(s3_uri)
    try:
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"NoSuchKey", "404", "NotFound"}:
            return {}
        raise
    return json.loads(obj["Body"].read().decode("utf-8"))




def read_candidate_metrics(model_dir):
    model_dir = Path(model_dir)
    direct_metrics = model_dir / "eval_metrics.json"
    if direct_metrics.exists():
        return json.loads(direct_metrics.read_text(encoding="utf-8"))

    archives = list(model_dir.rglob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"No model.tar.gz archive found under {model_dir}")

    with tarfile.open(archives[0], "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("eval_metrics.json"):
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                return json.loads(extracted.read().decode("utf-8"))

    raise FileNotFoundError("eval_metrics.json was not found in the model artifact")




def metric_value(metrics, metric_name):
    value = metrics.get(metric_name)
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if value is None:
        return None
    return float(value)




def main():
    if not CURRENT_METRICS_S3_URI:
        raise ValueError("CURRENT_METRICS_S3_URI environment variable is required")

    candidate_metrics = read_candidate_metrics(MODEL_DIR)
    previous_metrics = load_previous_metrics(CURRENT_METRICS_S3_URI)
    candidate = metric_value(candidate_metrics, METRIC_NAME)
    previous = metric_value(previous_metrics, METRIC_NAME)

    if candidate is None:
        raise ValueError(f"Candidate metrics do not contain {METRIC_NAME!r}: {candidate_metrics}")

    previous_for_compare = previous if previous is not None else -1.0
    should_deploy = int(candidate > previous_for_compare + MIN_DELTA)

    report = {
        "metrics": {
            METRIC_NAME: {"value": candidate},
            "candidate": candidate_metrics,
        },
        "previous": {
            METRIC_NAME: previous,
            "raw": previous_metrics,
        },
        "deployment": {
            "should_deploy": should_deploy,
            "reason": "candidate improved" if should_deploy else "candidate did not beat previous metric",
            "min_delta": MIN_DELTA,
        },
    }

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("Wrote evaluation report to %s: %s", output_path, report)


if __name__ == "__main__":
    main()
