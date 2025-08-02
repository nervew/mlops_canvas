from typing import List
import pandas as pd

class ITransformerValidator:
    def validate(
        self,
        transformer,
        df: pd.DataFrame,
        features_finales: List[str],
        verbose: bool = True,
    ):
        """Transforma df y devuelve (df_filtrado, lista_features_utilizadas)."""
        raise NotImplementedError