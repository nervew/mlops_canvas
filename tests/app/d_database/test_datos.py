import pandas as pd

from app.d_database.datos import generate_synthetic_patient_data


def test_generate_synthetic_patient_data_properties():
    df = generate_synthetic_patient_data()
    assert df.shape == (336, 6)
    assert list(df.columns) == ["Semana", "target", "EPS_A", "EPS_B", "EPS_C", "EPS_D"]
    assert pd.api.types.is_datetime64_any_dtype(df["Semana"])
    assert (df[["EPS_A", "EPS_B", "EPS_C", "EPS_D"]] >= 0).all().all()
    expected_target = df[["EPS_A", "EPS_B", "EPS_C", "EPS_D"]].sum(axis=1)
    pd.testing.assert_series_equal(df["target"], expected_target, check_names=False)
