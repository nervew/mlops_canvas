import time
from unittest import mock

import pytest
from botocore.exceptions import ClientError

from mlops.aws_registry_helper import AWSModelRegistryHelper


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "boom"}}, "op")


def test_register_model_creates_group_and_package(mocker):
    client = mocker.Mock()
    mocker.patch("boto3.client", return_value=client)
    client.describe_model_package_group.side_effect = _client_error("ResourceNotFound")
    client.create_model_package.return_value = {
        "ModelPackageArn": "arn:aws:sagemaker:::model-package/1"
    }

    helper = AWSModelRegistryHelper(region_name="us-east-1")
    arn = helper.register_model(
        model_data_url="s3://bucket/model.tar.gz",
        image_uri="1234.dkr.ecr.us-east-1.amazonaws.com/image:latest",
        model_package_group_name="Group",
        model_package_name="Model",
    )

    assert arn.endswith("model-package/1")
    client.create_model_package_group.assert_called_once_with(
        ModelPackageGroupName="Group", ModelPackageGroupDescription=""
    )
    client.create_model_package.assert_called_once()


def test_retry_logic(mocker):
    client = mocker.Mock()
    mocker.patch("boto3.client", return_value=client)
    client.describe_model_package_group.return_value = {}
    client.create_model_package.side_effect = [
        _client_error("ThrottlingException"),
        {"ModelPackageArn": "arn:aws:sagemaker:::model-package/2"},
    ]
    sleep = mocker.patch("time.sleep")

    helper = AWSModelRegistryHelper(region_name="us-east-1")
    arn = helper.register_model(
        model_data_url="s3://bucket/model.tar.gz",
        image_uri="1234.dkr.ecr.us-east-1.amazonaws.com/image:latest",
        model_package_group_name="Group",
    )

    assert arn.endswith("model-package/2")
    assert client.create_model_package.call_count == 2
    sleep.assert_called_once()
