# AWSModelRegistryHelper

The `AWSModelRegistryHelper` is a lightweight wrapper around the
[SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html).
It mirrors the ergonomics of `MLflowHelper` while focusing solely on model
registration.  The helper automatically creates model package groups when
required and retries transient failures.

```python
from mlops.aws_registry_helper import AWSModelRegistryHelper

helper = AWSModelRegistryHelper(region_name="us-east-1")
arn = helper.register_model(
    model_data_url="s3://my-bucket/model.tar.gz",
    image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
    model_package_group_name="MyGroup",
    model_package_name="MyModel",
)
print("Model registered:", arn)
```

The helper purposely exposes only a subset of the SageMaker API to keep the
interface clear and maintainable.  Additional parameters can be added as the
project requirements evolve.
