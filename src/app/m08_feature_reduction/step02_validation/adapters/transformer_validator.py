class TransformerValidator:
    def validate(self, transformer, df, features_finales):
        assert hasattr(transformer, 'fit') and hasattr(transformer, 'transform'), \
            "transformer debe implementar fit y transform"
        transformed = transformer.transform(df)
        missing = [f for f in features_finales if f not in transformed.columns]
        assert not missing, f"Columnas faltantes tras transformación: {missing}"
        return transformed[features_finales]
