# feature_engineering/domain/transformer.py
from dataclasses import dataclass, field
from pathlib import Path
import datetime as dt
import joblib
from sklearn.base import TransformerMixin

from feature_engineering.infrastructure.robust_fe import get_feature_names


@dataclass
class FeatureTransformer:
    """
    Envuelve el Pipeline sklearn y expone helpers + delega métodos de scikit-learn.
    """
    transformer: TransformerMixin
    created_at: str = field(
        default_factory=lambda: dt.datetime.utcnow().isoformat(timespec="seconds")
    )
    version: str = "0.1.0"

    # ---------- Persistencia -------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.transformer,
                "created_at": self.created_at,
                "version": self.version,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "FeatureTransformer":
        obj = joblib.load(path)
        return cls(
            transformer=obj["pipeline"],
            created_at=obj["created_at"],
            version=obj["version"],
        )

    # ---------- Helpers ------------------------------------------------------
    def get_output_feature_names(self) -> list[str]:
        preproc = self.transformer.named_steps["preprocess"]
        return get_feature_names(preproc)

    # ---------- Delegación a scikit-learn -----------------------------------
    def fit(self, X, y=None):
        """Ajusta el pipeline interno y devuelve self (estilo scikit-learn)."""
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        """Transforma sin volver a ajustar (para producción)."""
        return self.transformer.transform(X)

    def fit_transform(self, X, y=None):
        """Ajusta y transforma en una sola llamada."""
        return self.transformer.fit_transform(X, y)
