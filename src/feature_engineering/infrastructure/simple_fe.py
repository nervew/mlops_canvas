import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_transformer(df: pd.DataFrame) -> Pipeline:
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('target')
    scaler = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols)
    ], remainder='passthrough')
    return Pipeline([('preprocess', scaler)])
