import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from .custom_transformers import DateDecomposer


# ----------------------------------------------------------------------
# Construcción del pipeline
# ----------------------------------------------------------------------
def build_transformer(df: pd.DataFrame, *, target: str = "target") -> Pipeline:
    numeric_cols = (
        df.select_dtypes(include=["int64", "float64"])
        .columns.drop(target, errors="ignore")
        .to_list()
    )
    categorical_cols = (
        df.select_dtypes(include=["object", "category", "bool"]).columns.to_list()
    )
    date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.to_list()

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,  # scikit-learn >=1.4
                ),
            ),
        ]
    )

    date_pipe = (
        Pipeline([("decomposer", DateDecomposer()), ("scaler", StandardScaler())])
        if date_cols
        else "drop"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
            ("date", date_pipe, date_cols),
        ],
        remainder="drop",  # <<- evita que pase la columna target
    )

    return Pipeline([("preprocess", preprocessor)])


# ----------------------------------------------------------------------
# Helper para obtener nombres de salida
# ----------------------------------------------------------------------
def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Devuelve la lista completa de columnas después del transformador."""
    names: list[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if trans in ("drop", None):
            continue

        cols = list(cols)  # garantiza list

        if name == "cat":  # OneHotEncoder
            enc = trans.named_steps["encoder"]
            names.extend(enc.get_feature_names_out(cols))
        elif name == "date":
            for col in cols:
                names.extend([f"{col}_year", f"{col}_month", f"{col}_day"])
        else:  # num u otros
            names.extend(cols)
    return names
