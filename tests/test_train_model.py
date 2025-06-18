from hexagonal.adapters.toy_data_repository import ToyDataRepository
from hexagonal.adapters.local_model_repository import LocalModelRepository
from hexagonal.use_cases.train_model import TrainModelUseCase


def test_train_model_execution():
    data_repo = ToyDataRepository(n_samples=100)
    model_repo = LocalModelRepository(directory="test_models")
    use_case = TrainModelUseCase(data_repo, model_repo)
    result = use_case.execute()
    assert 0.0 <= result.accuracy <= 1.0
