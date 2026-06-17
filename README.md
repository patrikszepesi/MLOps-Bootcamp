# SageMaker Recycling Classifier: Real-Time Inference, Bedrock Feedback, and Metric-Gated Retraining

This repository contains an end-to-end SageMaker computer vision course lab for image classification.

The project trains a recycling waste classifier on the public Hugging Face dataset `viola77data/recycling-dataset`, deploys it to a SageMaker real-time endpoint, captures low-confidence predictions through API Gateway and Lambda, uses Amazon Bedrock Claude Sonnet 4.6 to label small low-confidence batches from a constrained class list, and starts a SageMaker Pipeline retraining job.

The production gate is explicit: the retraining pipeline deploys the candidate model only when its validation metric beats the currently deployed model.

This repo does not create IAM roles, IAM policies, Lambda functions, or API Gateway resources. Create those manually and use the Python files in `scripts/` as the Lambda and SageMaker entry points.

## Workflow

1. The notebook loads the public Hugging Face dataset for EDA.
2. The notebook stages that dataset to S3 as the training source.
3. The notebook trains the first image classifier in SageMaker from the S3 dataset input channel.
4. The notebook deploys the first real-time endpoint with data capture enabled.
5. API Gateway invokes Lambda with an image.
6. Lambda invokes the SageMaker endpoint.
7. If endpoint confidence is below the threshold, Lambda saves the image and prediction metadata to S3.
8. When three pending low-confidence examples exist, Lambda asks Bedrock Claude Sonnet 4.6 to choose exactly one label from the 11 allowed recycling labels.
9. Lambda writes Bedrock labels to S3 and starts the SageMaker Pipeline.
10. The pipeline retrains on the S3-staged base dataset plus the Bedrock-labeled examples.
11. The pipeline compares candidate validation accuracy to the current deployed model accuracy.
12. The endpoint is updated only if the candidate is better.

## Main Notebook

Start here:

[`notebooks/01_sagemaker_recycling_classifier_bedrock_retraining.ipynb`](notebooks/01_sagemaker_recycling_classifier_bedrock_retraining.ipynb)

## Dataset

Hugging Face dataset: [`viola77data/recycling-dataset`](https://huggingface.co/datasets/viola77data/recycling-dataset)

Classes:

- `aluminium`
- `batteries`
- `cardboard`
- `disposable plates`
- `glass`
- `hard plastic`
- `paper`
- `paper towel`
- `polystyrene`
- `soft plastics`
- `takeaway cups`

## Bedrock Model

The Lambda defaults to the global Amazon Bedrock model ID:

```text
global.anthropic.claude-sonnet-4-6
```

If your account requires a regional inference profile, override `BEDROCK_MODEL_ID` in the notebook before deploying the Lambda.

## Manual AWS Resources

Create these outside the notebook:

- API Lambda using `scripts/lambda_api_handler.py` with handler `lambda_api_handler.lambda_handler`
- deployment Lambda using `scripts/lambda_deploy_model.py` with handler `lambda_deploy_model.lambda_handler`
- API Gateway route such as `POST /classify` integrated with the API Lambda
- IAM role/policies for the API Lambda to invoke SageMaker Runtime, invoke Bedrock, read/write S3, and start the SageMaker Pipeline
- IAM role/policies for the deployment Lambda to create/update SageMaker model, endpoint config, and endpoint, plus write current metrics to S3
- SageMaker execution role permission to invoke the deployment Lambda from the pipeline Lambda step

The notebook prints the API Lambda environment variables to copy into your manually-created Lambda.

## Hard-Coded Training Settings

The SageMaker scripts do not parse command-line arguments. Course settings such as dataset, model, batch sizes, sample limits, and metric name are constants inside the Python files. The only dynamic retraining value is `LowConfidenceS3Uri`, because each low-confidence batch has a new S3 prefix.

## Production Note

Bedrock labels are automated feedback, not human ground truth. This is appropriate for a course demo and for low-risk assisted labeling loops. In production, log Bedrock confidence, audit labels, and require human review for costly or regulated decisions.
