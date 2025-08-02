from typing import List
import pandas as pd
from ..ports.validator import ITransformerValidator

class TransformerValidator(ITransformerValidator):
    def validate(
        self,
        transformer,
        df: pd.DataFrame,
        features_finales: List[str],
        verbose: bool = True,
    ):
        transformed = transformer.transform(df)
        present = set(transformed.columns)
        missing = [f for f in features_finales if f not in present]
        usable = [f for f in features_finales if f in present]
        return transformed[usable], usable
