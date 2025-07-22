import pandas as pd
import numpy as np
import json
import re
from typing import Dict, Any, Optional
import os  # ← IMPORTANTE

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

class FeatureValidator:
    def __init__(self):
        self.profile: Dict[str, Any] = {}
        self.null_threshold = 0.05
        self.unique_threshold = 0.4 # 40% 
    
    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None, max_categorias: int = 50):
        self.profile = {}
        for col in df.columns:
            series = df[col]
            col_profile = {}
            col_profile['dtype'] = str(series.dtype)
            col_profile['null_ratio'] = series.isnull().mean()
            col_profile['non_null_ratio'] = 1 - col_profile['null_ratio']
            unique_vals = series.dropna().unique()
            n_unique = len(unique_vals)
            col_profile['unique_count'] = n_unique
            col_profile['unique_ratio'] = n_unique / max(len(series.dropna()), 1)
            col_profile['is_constant'] = series.dropna().nunique() <= 1
            
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                col_profile['min'] = series.min()
                col_profile['max'] = series.max()
                col_profile['mean'] = series.mean()
                col_profile['median'] = series.median()
                col_profile['std'] = series.std()
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                col_profile['outlier_ratio'] = len(outliers) / max(len(series.dropna()), 1)
                col_profile['iqr_lower_bound'] = lower_bound
                col_profile['iqr_upper_bound'] = upper_bound
                col_profile['is_monotonic_increasing'] = series.is_monotonic_increasing
                col_profile['is_monotonic_decreasing'] = series.is_monotonic_decreasing
            
            elif pd.api.types.is_string_dtype(series) or series.dtype == 'object':
                series_str = series.dropna().astype(str)
                lengths = series_str.map(len)
                col_profile['min_length'] = int(lengths.min()) if not lengths.empty else None
                col_profile['max_length'] = int(lengths.max()) if not lengths.empty else None
                if n_unique <= max_categorias:
                    email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
                    sample_values = series.dropna().sample(min(100, len(series.dropna())))
                    email_matches = sample_values.str.match(email_pattern).sum()
                    col_profile['regex_pattern'] = email_pattern if email_matches > 0.8 * len(sample_values) else None
                    def has_invalid_chars(s):
                        try:
                            s.encode('ascii')
                            return False
                        except UnicodeEncodeError:
                            return True
                    col_profile['invalid_chars_ratio'] = series.dropna().map(has_invalid_chars).mean()
                    col_profile['category_counts'] = series.value_counts(dropna=True).to_dict()
                else:
                    col_profile['regex_pattern'] = None
                    col_profile['invalid_chars_ratio'] = None
                    col_profile['category_counts'] = None
            
            elif pd.api.types.is_bool_dtype(series):
                col_profile['true_ratio'] = series.mean()
                col_profile['false_ratio'] = 1 - col_profile['true_ratio']
            
            elif pd.api.types.is_datetime64_any_dtype(series):
                col_profile['min_date'] = str(series.min()) if not series.isnull().all() else None
                col_profile['max_date'] = str(series.max()) if not series.isnull().all() else None
            
            else:
                col_profile['sample_values'] = series.dropna().sample(min(10, len(series.dropna()))).tolist()
            
            col_profile['has_duplicates'] = series.duplicated().any()
            self.profile[col] = col_profile
        return self.profile
    
    def save_profile(self, filepath: str):
        # Asegura el directorio antes de guardar
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(convert_numpy_types(self.profile), f, indent=4)
    
    def load_profile(self, filepath: str):
        with open(filepath, 'r') as f:
            self.profile = json.load(f)
    
    def validate(self, df: pd.DataFrame, reporte_path: Optional[str] = None, max_issues_report: int = 10) -> Dict[str, Any]:
        report = {'valido': True, 'detalles': {}}
        for col, reglas in self.profile.items():
            col_report = {'aprobado': True, 'problemas': []}
            if col not in df.columns:
                col_report['aprobado'] = False
                col_report['problemas'].append('Columna ausente')
                report['valido'] = False
                report['detalles'][col] = col_report
                continue
            
            series = df[col]
            actual_dtype = str(series.dtype)
            expected_dtype = reglas['dtype']
            if actual_dtype != expected_dtype:
                col_report['aprobado'] = False
                col_report['problemas'].append(f'Tipo de dato esperado {expected_dtype}, pero se encontró {actual_dtype}')
            
            null_ratio = series.isnull().mean()
            if null_ratio > reglas.get('null_ratio', 1.0) + 0.01:
                col_report['aprobado'] = False
                col_report['problemas'].append(f'Proporción de valores nulos {null_ratio:.2%} supera la aprendida {reglas.get("null_ratio", 0):.2%}')
            
            non_null_ratio = 1 - null_ratio
            if non_null_ratio < self.null_threshold:
                col_report['aprobado'] = False
                col_report['problemas'].append(f'Proporción de valores no nulos {non_null_ratio:.2%} menor al mínimo permitido {self.null_threshold:.2%}')
            
            unique_ratio = series.dropna().nunique() / max(len(series.dropna()), 1)
            if unique_ratio > self.unique_threshold:
                col_report['aprobado'] = False
                col_report['problemas'].append(f'Proporción de valores únicos {unique_ratio:.2%} supera el máximo permitido {self.unique_threshold:.2%}')
            
            if series.dropna().nunique() <= 1 and not reglas.get('is_constant', False):
                col_report['aprobado'] = False
                col_report['problemas'].append('Columna constante pero no lo era en el entrenamiento')
            
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                min_val = series.min()
                max_val = series.max()
                expected_min = reglas.get('min')
                expected_max = reglas.get('max')
                if expected_min is not None and min_val < expected_min:
                    col_report['aprobado'] = False
                    col_report['problemas'].append(f'Valor mínimo {min_val} menor al aprendido {expected_min}')
                if expected_max is not None and max_val > expected_max:
                    col_report['aprobado'] = False
                    col_report['problemas'].append(f'Valor máximo {max_val} mayor al aprendido {expected_max}')
                iqr_low = reglas.get('iqr_lower_bound')
                iqr_up = reglas.get('iqr_upper_bound')
                if iqr_low is not None and iqr_up is not None:
                    outliers = series[(series < iqr_low) | (series > iqr_up)]
                    outlier_ratio = len(outliers) / max(len(series.dropna()), 1)
                    if outlier_ratio > reglas.get('outlier_ratio', 0) + 0.01:
                        col_report['aprobado'] = False
                        col_report['problemas'].append(f'Proporción de outliers {outlier_ratio:.2%} supera la aprendida {reglas.get("outlier_ratio", 0):.2%}')
                if reglas.get('is_monotonic_increasing') and not series.is_monotonic_increasing:
                    col_report['aprobado'] = False
                    col_report['problemas'].append('Columna no es monotónica creciente como en el entrenamiento')
                if reglas.get('is_monotonic_decreasing') and not series.is_monotonic_decreasing:
                    col_report['aprobado'] = False
                    col_report['problemas'].append('Columna no es monotónica decreciente como en el entrenamiento')
            elif pd.api.types.is_string_dtype(series) or series.dtype == 'object':
                series_str = series.dropna().astype(str)
                lengths = series_str.map(len)
                min_length = reglas.get('min_length')
                max_length = reglas.get('max_length')
                if min_length is not None and lengths.min() < min_length:
                    col_report['aprobado'] = False
                    col_report['problemas'].append(f'Longitud mínima de cadena {lengths.min()} menor a la aprendida {min_length}')
                if max_length is not None and lengths.max() > max_length:
                    col_report['aprobado'] = False
                    col_report['problemas'].append(f'Longitud máxima de cadena {lengths.max()} mayor a la aprendida {max_length}')
                
                regex = reglas.get('regex_pattern')
                if regex is not None:
                    print(regex)
                    def cumple_regex(x):
                        if not isinstance(x, str):
                            print(False)
                            return False
                        return bool(re.fullmatch(regex, x.strip()))
                    invalid_regex = series.dropna().map(lambda x: not cumple_regex(x)).sum()
                    if invalid_regex > 0:
                        col_report['aprobado'] = False
                        col_report['problemas'].append(f'{invalid_regex} valores no cumplen patrón regex')
                def has_invalid_chars(s):
                    if not isinstance(s, str):
                        return False
                    try:
                        s.encode('ascii')
                        return False
                    except UnicodeEncodeError:
                        return True
                invalid_chars_ratio = series.dropna().map(has_invalid_chars).mean()
                invalid_chars_ratio_aprendido = reglas.get('invalid_chars_ratio')
                if invalid_chars_ratio_aprendido is None:
                    invalid_chars_ratio_aprendido = 0.0

                if invalid_chars_ratio > invalid_chars_ratio_aprendido + 0.01:
                    col_report['aprobado'] = False
                    col_report['problemas'].append(f'Proporción de caracteres inválidos {invalid_chars_ratio:.2%} supera la aprendida {invalid_chars_ratio_aprendido:.2%}')
                learned_counts = reglas.get('category_counts', {})
                if learned_counts is not None:
                    current_counts = series.value_counts(dropna=True).to_dict()
                    for cat, learned_count in learned_counts.items():
                        current_count = current_counts.get(cat, 0)
                        if current_count < 0.5 * learned_count:
                            col_report['aprobado'] = False
                            col_report['problemas'].append(f'Categoría "{cat}" disminuyó de {learned_count} a {current_count}')
                    learned_cats = set(learned_counts.keys())
                    current_cats = set(current_counts.keys())
                    missing_cats = learned_cats - current_cats
                    if missing_cats:
                        col_report['aprobado'] = False
                        col_report['problemas'].append(f'Faltan categorías: {missing_cats}')
            elif pd.api.types.is_datetime64_any_dtype(series):
                min_date = pd.to_datetime(reglas.get('min_date'))
                max_date = pd.to_datetime(reglas.get('max_date'))
                if min_date and series.min() < min_date:
                    col_report['aprobado'] = False
                    col_report['problemas'].append(f'Fecha mínima {series.min()} anterior a la aprendida {min_date}')
                if max_date and series.max() > max_date:
                    col_report['aprobado'] = False
                    col_report['problemas'].append(f'Fecha máxima {series.max()} posterior a la aprendida {max_date}')
            if series.duplicated().any() and not reglas.get('has_duplicates', True):
                col_report['aprobado'] = False
                col_report['problemas'].append('Se encontraron duplicados no permitidos')
            if not col_report['aprobado']:
                report['valido'] = False
            
            report['detalles'][col] = col_report
        
        if reporte_path:
            # Asegura el directorio antes de guardar el reporte
            os.makedirs(os.path.dirname(reporte_path), exist_ok=True)
            with open(reporte_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=4)
        
        return report
