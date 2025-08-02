from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .step01_ingesta.adapters.parquet_loader import ParquetPartitionLoader
from .step02_validation.adapters.transformer_validator import TransformerValidator
from .step03_reduction.adapters.reducer import Reducer
from .step04_shap.adapters.shap_explainer import ShapExplainer
from .step05_pipeline.adapters.pipeline_builder import PipelineBuilder
from .step06_evaluation.adapters.evaluator import Evaluator
from .step07_export.adapters.exporter import Exporter


# ──────────────  Localizador robusto de 'data/raw/partitioned'  ───────────
def find_partition_dir(start: Path) -> Path:
    d = start.resolve()
    while d != d.parent:
        cand = d / "data" / "raw" / "partitioned"
        if (cand / "train_df.parquet").exists():
            return cand
        d = d.parent
    raise FileNotFoundError("No se encontró data/raw/partitioned/*_df.parquet")


THIS_FILE = Path(__file__).resolve()
PART_DIR  = find_partition_dir(THIS_FILE)
PROJECT_ROOT = PART_DIR.parent.parent.parent   # …/mlops_canvas
print(f"Buscando particiones en: {PART_DIR}")
# ──────────────────────────────────────────────────────────────────────────


def run_feature_reduction(
    transformer: Any,
    features:    List[str],
    params:      Optional[Dict[str, Any]] = None,
    use_reduction: bool = False,
    temporal_vars: Optional[List[str]] = None,
    target_var: str = "target",
    mae_anterior: float | None = None,   # ← opcional: pasa el MAE de tu modelo baseline
) -> Dict[str, Any]:
    """Orquesta el módulo de reducción y exporta métricas/SHAP/MAE."""
    temporal_vars = temporal_vars or []
    params = params or {"method": "pca", "n_components": 3}

    # 1) Ingesta
    loader = ParquetPartitionLoader(str(PART_DIR))
    X_tr_raw, X_te_raw, X_bk_raw = loader.load()

    # 2) Separar temporales + target
    def split(df: pd.DataFrame):
        temp = df[temporal_vars] if temporal_vars else pd.DataFrame(index=df.index)
        tgt  = df[[target_var]]  if target_var in df else pd.DataFrame(index=df.index)
        core = df.drop(columns=temporal_vars + [target_var], errors="ignore")
        return core, temp, tgt

    tr_core, tr_temp, tr_tgt = split(X_tr_raw)
    te_core, te_temp, te_tgt = split(X_te_raw)
    bk_core, bk_temp, bk_tgt = split(X_bk_raw)

    # 3) Transformador inicial
    tr_full = transformer.transform(tr_core)
    te_full = transformer.transform(te_core)
    bk_full = transformer.transform(bk_core)

    # 4) Validación de features
    validator = TransformerValidator()
    tr_ready, usable_feats = validator.validate(transformer, tr_core, features, verbose=True)
    te_ready = te_full[usable_feats]
    bk_ready = bk_full[usable_feats]

    # 5) Reductor
    reducer = Reducer(
        method=params.get("method", "pca") if use_reduction and len(usable_feats) > 1 else None,
        n_components=params.get("n_components", 3),
        random_state=params.get("random_state", 42),
    )
    tr_red = reducer.fit_transform(tr_ready)
    te_red = reducer.transform(te_ready)
    bk_red = reducer.transform(bk_ready)

    # 6) SHAP
    shap_exp = ShapExplainer(
        explainer_type=params.get("explainer_type", "kernel"),
        background_size=params.get("background_size", 100),
        sample_size=params.get("sample_size", 100),
        random_state=params.get("random_state", 42),
    )
    shap_vals, shap_summary = shap_exp.explain(
        reducer, pd.DataFrame(tr_ready, columns=usable_feats)
    )

    # 7) Pipeline final
    builder = PipelineBuilder()
    pipeline_final = builder.build(transformer, usable_feats, reducer)

    # 8) Evaluación   ------------------------------------------------------
    evaluator = Evaluator()
    metrics = evaluator.evaluate(
        pipeline_final, te_ready, te_tgt[target_var]
    ) if not te_tgt.empty else {}

    # calculamos MAE para comparar
    y_true = te_tgt[target_var] if target_var in te_tgt else None
    mae_reduccion = (
        mean_absolute_error(y_true, pipeline_final.predict(te_ready))
        if (y_true is not None and hasattr(pipeline_final, "predict"))
        else None
    )

    # 9) Reconstruir DataFrames -------------------------------------------
    def restore(mat, temp, tgt):
        df_out = pd.DataFrame(mat, index=temp.index)
        if not temp.empty:
            df_out[temp.columns] = temp
        if not tgt.empty:
            df_out[tgt.columns] = tgt
        return df_out

    tr_export = restore(tr_red, tr_temp, tr_tgt)
    te_export = restore(te_red, te_temp, te_tgt)
    bk_export = restore(bk_red, bk_temp, bk_tgt)

    # 10) Column-importance (SHAP)  ----------------------------------------
    # Aseguramos un vector 1-D
    shap_summary_vec = np.asarray(shap_summary).ravel()

    col_imp = sorted(
        [
        {"feature": f, "importance": float(i)}
        for f, i in zip(usable_feats, shap_summary_vec)
        ],
        key=lambda d: d["importance"],
        reverse=True,
    )


    # 11) Exportar artefactos  --------------------------------------------
    exporter = Exporter()
    exporter.export_all(
        transformer=pipeline_final,
        shap_values=shap_vals,
        shap_summary=shap_summary,
        dfs={"train": tr_export, "test": te_export, "backtest": bk_export},
        metrics_json={
            "mae_anterior": mae_anterior,
            "mae_reduccion": mae_reduccion,
            "delta_mae": None if mae_reduccion is None or mae_anterior is None
                         else mae_reduccion - mae_anterior,
            "column_importance": col_imp,
            **metrics,   # agrega otras métricas que devuelva el Evaluator
        },
    )

    # Log en consola
    #print(f"Train shape: {tr_export.shape}")
    #print(f"Test  shape: {te_export.shape}")
    #print(f"Back  shape: {bk_export.shape}")
    if mae_reduccion is not None:
        print(f"MAE reducción: {mae_reduccion:.4f} "
              f"(Δ = {mae_reduccion - mae_anterior:.4f}" if mae_anterior else "", ")")

    return {
        "transformer": pipeline_final,
        "metrics": metrics,
        "shap_summary": shap_summary,
        "mae_reduccion": mae_reduccion,
        "column_importance": col_imp,
    }
