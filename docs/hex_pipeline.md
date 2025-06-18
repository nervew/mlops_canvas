# Diagrama de flujo - Pipeline Hexagonal

```mermaid
flowchart TD
    A[DataRepository] --> B[TrainModelUseCase]
    B --> C[Split Data]
    C --> D[Feature Engineering]
    D --> E[Feature Selection]
    E --> F[Model Training]
    F --> G[ModelRepository]
    B --> H[Evaluate Model]
```
