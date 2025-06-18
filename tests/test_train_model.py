from hexagonal.adapters.toy_data_repository import ToyDataRepository
from hexagonal.adapters.local_model_repository import LocalModelRepository
from hexagonal.use_cases.train_model import TrainModelUseCase


def test_train_model_classification():
    data_repo = ToyDataRepository(n_samples=100, task_type="classification")
    model_repo = LocalModelRepository(directory="test_models_clf")
    use_case = TrainModelUseCase(data_repo, model_repo, task_type="classification")
    result = use_case.execute()
    assert 0.0 <= result.score <= 1.0


def test_train_model_regression():
    data_repo = ToyDataRepository(n_samples=100, task_type="regression")
    model_repo = LocalModelRepository(directory="test_models_reg")
    use_case = TrainModelUseCase(data_repo, model_repo, task_type="regression")
    result = use_case.execute()
    assert isinstance(result.score, float)
