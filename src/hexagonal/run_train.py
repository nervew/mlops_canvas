from hexagonal.adapters.toy_data_repository import ToyDataRepository
from hexagonal.adapters.local_model_repository import LocalModelRepository
from hexagonal.use_cases.train_model import TrainModelUseCase


def main() -> None:
    data_repo = ToyDataRepository()
    model_repo = LocalModelRepository()
    use_case = TrainModelUseCase(data_repo, model_repo)
    result = use_case.execute()
    print(f"Accuracy: {result.accuracy:.4f}")


if __name__ == "__main__":
    main()
