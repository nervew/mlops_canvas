import pandas as pd
import great_expectations as gx
import json
from pathlib import Path

# Setup de rutas
BASE_DIR = Path(__file__).resolve().parent
GX_DIR   = BASE_DIR / "gx"
EXP_DIR  = GX_DIR / "expectations"
EXP_DIR.mkdir(parents=True, exist_ok=True)

def suite_name(ds):
    return f"{ds}_suite"

def generate_suite(df, dataset, gx_dir=None):
    name = suite_name(dataset)
    path = EXP_DIR / f"{name}.json"
    path.unlink(missing_ok=True)
    suite = {
        "name": name,
        "expectations": [
            {"type": "expect_table_row_count_to_be_between", "kwargs": {"min_value": 1}},
            {"type": "expect_column_values_to_not_be_null", "kwargs": {"column": "a"}}
        ],
        "meta": {}
    }
    path.write_text(json.dumps(suite, indent=2))
    print(f"✅ Suite guardado en {path}")
    return path

def validate_dataframe(df, dataset, gx_dir=None, raise_error=True):
    ctx  = gx.get_context(context_root_dir=str(GX_DIR))
    name = suite_name(dataset)
    validator = ctx.get_validator(df, expectation_suite_name=name)
    result = validator.validate()
    print("¿Validación exitosa?", result["success"])
    if not result["success"] and raise_error:
        raise ValueError(f"Validación falló para {name}")
    return result
