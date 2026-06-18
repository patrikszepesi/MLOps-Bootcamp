import json
import os
import re
import time
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

sm = boto3.client("sagemaker")
s3 = boto3.client("s3")


def _safe_name(prefix):
    """Create a unique SageMaker resource name from a readable prefix.
    This prevents model and endpoint-config name collisions every time the pipeline deploys a candidate.
    """
    cleaned = re.sub(r"[^A-Za-z0-9-]", "-", prefix).strip("-")
    suffix = str(int(time.time()))
    max_prefix = 63 - len(suffix) - 1
    return f"{cleaned[:max_prefix]}-{suffix}"


def _parse_s3_uri(uri):
    """Split an S3 URI into the bucket and object key that boto3 needs.
    This is used when writing the current deployed model's metric record back to S3."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _endpoint_exists(endpoint_name):
    """Check whether the production endpoint already exists.
    The deploy Lambda uses this to choose between creating the endpoint once or updating it later.
    """
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code == "ValidationException":
            return False
        raise


def _put_current_metrics(uri, payload):
    """Write the new deployed model's metrics and metadata to S3.
    Future pipeline runs compare against this record before deciding whether to deploy a new candidate.
    """
    bucket, key = _parse_s3_uri(uri)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def lambda_handler(event, context):
    """Deploy a pipeline-approved model candidate to the real-time endpoint.
    This creates a new SageMaker model and endpoint config, then creates or updates the stable endpoint name.
    """
    endpoint_name = event["endpoint_name"]
    model_data = event["model_data"]
    role_arn = event["role_arn"]
    image_uri = event["inference_image_uri"]
    instance_type = event.get("instance_type", "ml.m5.large")
    data_capture_s3_uri = event["data_capture_s3_uri"]
    current_metrics_s3_uri = event["current_metrics_s3_uri"]
    candidate_accuracy = float(event["candidate_accuracy"])
    region = os.environ.get("AWS_REGION", boto3.session.Session().region_name)

    model_name = _safe_name(f"{endpoint_name}-model")
    endpoint_config_name = _safe_name(f"{endpoint_name}-config")

    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_REGION": region,
            },
        },
    )

    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": data_capture_s3_uri,
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            "CaptureContentTypeHeader": {"JsonContentTypes": ["application/json"]},
        },
    )

    if _endpoint_exists(endpoint_name):
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        status = "update-started"
    else:
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        status = "create-started"

    _put_current_metrics(
        current_metrics_s3_uri,
        {
            "accuracy": candidate_accuracy,
            "model_data": model_data,
            "model_name": model_name,
            "endpoint_name": endpoint_name,
            "endpoint_config_name": endpoint_config_name,
            "status": status,
            "updated_at_epoch": int(time.time()),
        },
    )

    return {
        "status": status,
        "endpoint_name": endpoint_name,
        "model_name": model_name,
    }
