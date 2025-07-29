# m08_feature_reduction/pipeline_reduction.py

from pathlib import Path
import pandas as pd

from .step01_ingesta.adapters.parquet_loader import ParquetPartitionLoader
from .step02_validation.adapters.transformer_validator import TransformerValidator
from .step03_reduction.adapters.reducer import Reducer
from .step04_shap.adapters.shap_explainer import ShapExplainer
from .step05_pipeline.adapters.pipeline_builder import PipelineBuilder
from .step06_evaluation.adapters.evaluator import Evaluator
from .step07_export.adapters.exporter import Exporter


def run_feature_reduction(
    transformer,
    features: list[str],
    params: dict,
    use_reduction: bool = False,
    temporal_vars: list[str] = ["Semana"],
    target_var: str = "target"
) -> dict:
    """
    Pipeline robusto de reducción dimensional con:
      - separación/restauración automática de variables temporales y target
      - filtrado de features inconsistentes
      - ajuste dinámico de n_components para PCA
      - cálculo de SHAP values
      - evaluación (reconstrucción o predict)
      - exportación de artefactos
    """

    # 1) INGESTA
    project_src = Path(__file__).resolve().parents[2]   # mlops_canvas/src
    parquet_dir = project_src / "data" / "raw" / "partitioned"
    loader = ParquetPartitionLoader(str(parquet_dir))
    train, test, back = loader.load()

    # 2) SEPARAR temporales y target
    def separate(df: pd.DataFrame):
        temp = df[temporal_vars].copy()
        tgt  = df[[target_var]].copy()
        feats = df.drop(columns=temporal_vars + [target_var])
        return feats, temp, tgt

    X_train, train_temp, train_tgt = separate(train)
    X_test,  test_temp,  test_tgt  = separate(test)
    X_back,  back_temp,  back_tgt  = separate(back)

    # 3) FILTRAR features para descartar temporales/target
    features_clean = [f for f in features if f not in temporal_vars + [target_var]]

    # 4) VALIDACIÓN y TRANSFORMACIÓN
    validator = TransformerValidator()
    X_train_ready = validator.validate(transformer, X_train, features_clean)
    X_test_ready  = transformer.transform(X_test)[features_clean]
    X_back_ready  = transformer.transform(X_back)[features_clean]

    # 5) REDUCCIÓN dimensional (PCA o nada)
    n_feats = X_train_ready.shape[1]
    requested_n = params.get("n_components", 3)
    n_comp = min(requested_n, n_feats) if use_reduction else None

    reducer = Reducer(
        method=None if not use_reduction else params.get("method", "pca"),
        n_components=n_comp,
        random_state=params.get("random_state", 42)
    )
    reducer.fit(X_train_ready)

    # 6) SHAP values
    shap_exp = ShapExplainer(
        explainer_type=params.get("explainer_type", "kernel"),
        background_size=params.get("background_size", 100),
        sample_size=params.get("sample_size", 100),
        random_state=params.get("random_state", 42)
    )
    shap_values, shap_summary = shap_exp.explain(reducer, X_train_ready)

    # 7) BUILD final pipeline
    builder = PipelineBuilder()
    pipeline_final = builder.build(transformer, features_clean, reducer)

    # 8) EVALUACIÓN
    evaluator = Evaluator()
    metrics = evaluator.evaluate(pipeline_final, X_test_ready, test_tgt[target_var])

    # 9) RECONSTRUIR DataFrames con temporales + componentes + target
    def reconstruct(arr, temp_df, tgt_df):
        comps = pd.DataFrame(
            arr,
            index=temp_df.index,
            columns=[f"component_{i+1}" for i in range(arr.shape[1])]
        )
        return pd.concat([temp_df, comps, tgt_df], axis=1)

    train_final = reconstruct(reducer.transform(X_train_ready), train_temp, train_tgt)
    test_final  = reconstruct(reducer.transform(X_test_ready), test_temp, test_tgt)
    back_final  = reconstruct(reducer.transform(X_back_ready), back_temp, back_tgt)

    # 10) EXPORTAR artefactos
    exporter = Exporter()
    exporter.export_all(
        transformer=pipeline_final,
        shap_values=shap_values,
        shap_summary=shap_summary,
        dfs={"train": train_final, "test": test_final, "backtest": back_final}
    )

    return {
        "transformer": pipeline_final,
        "metrics": metrics,
        "shap_summary": shap_summary
    }
