import sys
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from melitk import logging
from melitk.fda2 import runtime
from meligenesis.artifacts.manager import GenesisArtifactManager as gam
from app.select_model.genesis.automl_manager.flaml_wrapper import FLAMLWrapper
# from app.select_model.genesis.automl_manager.mljar_wrapper import MLJARWrapper

logger = logging.getLogger("Model_Selector")

def validate_params(params: Dict[str, Any], keys: List[str]) -> None:
    """Ensure that each key in `required_keys` exists and is truthy in `params`."""
    missing = [k for k in keys if not params.get(k)]
    if missing:
        for key in missing:
            logger.info(f"❌ Missing required parameter: '{key}'.")
        sys.exit(1)

def extract_list(metadata: dict, key: str, single: bool = False) -> Union[Optional[Any], List[Any]]:
    raw = metadata.get(key)
    if raw is None:
        # In case someone explicitly set key: null
        logger.info(f"❌ Metadata['{key}'] is null; treating as empty list.")
        raw = []

    if isinstance(raw, str):
        raw = [raw]

    if not isinstance(raw, list):
        raise ValueError(f"❌ Expected metadata['{key}'] to be a list, got {type(raw)}")

    if single:
        if isinstance(raw, list):
            if not raw:
                logger.info(f"❌ No se encontraron valores para '{key}'; devolviendo None.")
                return None
            return raw[0]
        return raw

    if not isinstance(raw, list):
        return [raw]
    
    return raw

def create_artifacts(specs: List[Dict[str, Any]]) -> None:
    """Create and upload each artifact defined in specs."""
    for spec in specs:
        art = gam.create_artifact(**spec)
        art.upload_content()

def load_artifact_df(artifact_id: str) -> pd.DataFrame:
    """Fetch artifact by ID and return its .content as a DataFrame or dict."""
    art = gam.get_artifact_by_id(artifact_id=artifact_id)
    return art.content

def main() -> None:
    """Main entrypoint for the model selection task."""
    if runtime is None:
        logger.info("❌ Runtime is not available.")
        sys.exit(1)

    params = runtime.inputs.parameters or {}
    
    required = ["genesis_version", "train_set_id_artifact", "test_set_id_artifact", "metadata_id_artifact"]
    validate_params(params, required)

    # Load metadata and data
    version  = params["genesis_version"]
    train_id = params["train_set_id_artifact"]
    test_id  = params["test_set_id_artifact"]
    features_id = params["metadata_id_artifact"]

    features = load_artifact_df(features_id)
    target = extract_list(features, "target", single=True)
    columns = extract_list(features, "all")

    df_train = load_artifact_df(train_id)
    df_test = load_artifact_df(test_id)

    for df, name in ((df_train, "train"), (df_test, "test")):
        if target not in df.columns:
            logger.info(f"❌ Target column '{target}' not found in {name} set.")
            sys.exit(1)
        if df[target].isnull().any() or df[target].isna().any():
            logger.info(f"❌ Nulls detected in target column '{target}' of {name} set.")
            sys.exit(1)

    X_train = df_train[columns]
    y_train = df_train[target]

    X_test = df_test[columns]
    y_test = df_test[target]

    selector = FLAMLWrapper()
    selector.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    ranking     = selector.get_model_ranking()
    best_model  = selector.get_best_model()
    best_params = selector.get_best_params()

    specs = [
        {
            "name": "genesis_models_importance",
            "artifact_type": "fda.Bytes",
            "version": version,
            "content": ranking,
        },
        {
            "name": "genesis_selected_model",
            "artifact_type": "fda.Model",
            "version": version,
            "content": best_model[1],
        },
        {
            "name": "genesis_hiperparameters_selected_model",
            "artifact_type": "fda.Bytes",
            "version": version,
            "content": best_params,
        },
    ]
    create_artifacts(specs)

    try:
        imp_df = pd.DataFrame(ranking)
        logger.info("📊 Model ranking:\n" + imp_df.to_string(index=False))
    except Exception:
        logger.info("📊 Model ranking:\n" + str(ranking))

    logger.info(f"✅ Selected model: {best_model[0]}")
    logger.info(f"🔧 Hyperparameters: {best_params}")

if __name__ == "__main__":
    main()