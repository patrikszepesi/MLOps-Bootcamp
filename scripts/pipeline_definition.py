import os

import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker.lambda_helper import Lambda
from sagemaker.processing import ProcessingInput
from sagemaker.processing import ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.lambda_step import LambdaOutput
from sagemaker.workflow.lambda_step import LambdaOutputTypeEnum
from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.steps import TrainingStep







PROJECT = "recycling-classifier-course"
PIPELINE_NAME = f"{PROJECT}-pipeline"
ENDPOINT_NAME = f"{PROJECT}-endpoint"
MODEL_ID = "google/vit-base-patch16-224-in21k"
DATASET_ID = "viola77data/recycling-dataset"
TRAINING_INSTANCE_TYPE = "ml.g4dn.xlarge"
PROCESSING_INSTANCE_TYPE = "ml.m5.large"
DEPLOY_INSTANCE_TYPE = "ml.m5.large"
TRANSFORMERS_VERSION = "4.36.0"
PYTORCH_VERSION = "2.1.0"
PY_VERSION = "py310"


def get_pipeline(region, role, default_bucket, deploy_lambda_arn, inference_image_uri, scripts_dir=None):

    sagemaker_session = sagemaker.session.Session(default_bucket=default_bucket)
    pipeline_session = PipelineSession(
        boto_session=sagemaker_session.boto_session,
        sagemaker_client=sagemaker_session.sagemaker_client,
        default_bucket=default_bucket,
    )
    scripts_dir = scripts_dir or os.path.dirname(os.path.abspath(__file__))
    current_metrics_s3_uri = f"s3://{default_bucket}/{PROJECT}/model-registry/current_metrics.json"
    data_capture_s3_uri = f"s3://{default_bucket}/{PROJECT}/monitoring/data-capture"
    dataset_s3_uri = f"s3://{default_bucket}/{PROJECT}/datasets/recycling-hf-dataset"

    low_confidence_s3_uri = ParameterString("LowConfidenceS3Uri", default_value="")

    estimator = HuggingFace(
        entry_point="train.py",
        source_dir=scripts_dir,
        role=role,
        instance_count=1,
        instance_type=TRAINING_INSTANCE_TYPE,
        transformers_version=TRANSFORMERS_VERSION,
        pytorch_version=PYTORCH_VERSION,
        py_version=PY_VERSION,
        metric_definitions=[{"Name": "validation:accuracy", "Regex": "accuracy=([0-9\\.]+)"}],
        hyperparameters={"low_confidence_s3_uri": low_confidence_s3_uri},
        sagemaker_session=pipeline_session,
    )

    train_args = estimator.fit(inputs={"dataset": dataset_s3_uri})
    step_train = TrainingStep(name="TrainRecyclingClassifier", step_args=train_args)

    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_count=1,
        instance_type=PROCESSING_INSTANCE_TYPE,
        env={"CURRENT_METRICS_S3_URI": current_metrics_s3_uri},
        sagemaker_session=pipeline_session,
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    compare_args = processor.run(
        code=os.path.join(scripts_dir, "compare_model_metrics.py"),
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
            )
        ],
    )
    step_compare = ProcessingStep(
        name="CompareCandidateToCurrent",
        step_args=compare_args,
        property_files=[evaluation_report],
    )


    deploy_status = LambdaOutput(output_name="status", output_type=LambdaOutputTypeEnum.String)
    deployed_endpoint_name = LambdaOutput(output_name="endpoint_name", output_type=LambdaOutputTypeEnum.String)
    deployed_model_name = LambdaOutput(output_name="model_name", output_type=LambdaOutputTypeEnum.String)
    step_deploy = LambdaStep(
        name="DeployBetterCandidate",
        lambda_func=Lambda(function_arn=deploy_lambda_arn),
        inputs={
            "endpoint_name": ENDPOINT_NAME,
            "model_data": step_train.properties.ModelArtifacts.S3ModelArtifacts,
            "role_arn": role,
            "inference_image_uri": inference_image_uri,
            "instance_type": DEPLOY_INSTANCE_TYPE,
            "data_capture_s3_uri": data_capture_s3_uri,
            "current_metrics_s3_uri": current_metrics_s3_uri,
            "candidate_accuracy": JsonGet(
                step_name=step_compare.name,
                property_file=evaluation_report,
                json_path="metrics.accuracy.value",
            ),
        },
        outputs=[deploy_status, deployed_endpoint_name, deployed_model_name],
    )
    step_condition = ConditionStep(
        name="DeployOnlyIfAccuracyImproves",
        conditions=[
            ConditionEquals(
                left=JsonGet(
                    step_name=step_compare.name,
                    property_file=evaluation_report,
                    json_path="deployment.should_deploy",
                ),
                right=1,
            )
        ],
        if_steps=[step_deploy],
        else_steps=[],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[low_confidence_s3_uri],
        steps=[step_train, step_compare, step_condition],
        sagemaker_session=pipeline_session,
    )
