import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Callable, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, create_model
import importlib.util
from datetime import datetime
import os
import functools
from config import (
    API_TITLE, API_VERSION, MODEL_PATH, DATA_PATH, REQUIREMENTS_PATH,
    THRESHOLD, MODEL_VERSION, MODEL_NAME, OUTPUT_PREDICT_PROBA_KEY,
    OUTPUT_PREDICT_KEY, OUTPUT_THRESHOLD_KEY, ERROR_PREDICT_PROBA_VALUE,
    ERROR_PREDICT_VALUE, ERROR_THRESHOLD_VALUE, SHOW_THRESHOLD,
    ENVIRONMENT, BUILD_ID
)

app = FastAPI(title=API_TITLE, version=API_VERSION)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler personalizado que devuelve el detail directamente sin envolver en 'detail'"""
    if isinstance(exc.detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


model = None
input_columns = None
output_type = None
PredictRequest = None


def detect_framework() -> str:
    """Detecta el framework de ML desde requirements.txt"""
    if not REQUIREMENTS_PATH.exists():
        raise FileNotFoundError(f"No se encuentra {REQUIREMENTS_PATH}")
    
    with open(REQUIREMENTS_PATH, 'r') as f:
        requirements = f.read().lower()
    
    if 'scikit-learn' in requirements or 'sklearn' in requirements:
        return 'sklearn'
    elif 'xgboost' in requirements:
        return 'xgboost'
    elif 'lightgbm' in requirements:
        return 'lightgbm'
    elif 'catboost' in requirements:
        return 'catboost'
    elif 'tensorflow' in requirements:
        return 'tensorflow'
    elif 'torch' in requirements or 'pytorch' in requirements:
        return 'pytorch'
    else:
        return 'unknown'


def load_model():
    """Carga el modelo desde artifacts/model.pkl"""
    global model
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encuentra {MODEL_PATH}")
    
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
    except (ImportError, Exception):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        except (pickle.UnpicklingError, ValueError, TypeError):
            try:
                with open(MODEL_PATH, 'rb') as f:
                    unpickler = pickle.Unpickler(f)
                    unpickler.encoding = 'latin1'
                    model = unpickler.load()
            except Exception as e:
                raise ValueError(f"Error al cargar el modelo: {str(e)}. El archivo puede estar corrupto o ser incompatible.")
    
    return model


def infer_input_columns() -> List[str]:
    """Infiere las columnas de entrada desde el modelo y data.parquet"""
    global input_columns
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encuentra {DATA_PATH}")
    
    df = pd.read_parquet(DATA_PATH)
    
    if hasattr(model, 'feature_names_in_'):
        input_columns = list(model.feature_names_in_)
    elif hasattr(model, 'get_booster'):
        booster = model.get_booster()
        if hasattr(booster, 'feature_names'):
            input_columns = booster.feature_names
        else:
            input_columns = [f'feature_{i}' for i in range(len(df.columns))]
    elif hasattr(model, 'feature_importances_'):
        if hasattr(model, 'feature_names_in_'):
            input_columns = list(model.feature_names_in_)
        else:
            input_columns = list(df.columns)
    else:
        input_columns = list(df.columns)
    
    return input_columns


def infer_output_type() -> str:
    """Infiere el tipo de salida del modelo"""
    global output_type
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encuentra {DATA_PATH}")
    
    df = pd.read_parquet(DATA_PATH)
    
    sample_data = df[input_columns].iloc[:1] if input_columns else df.iloc[:1]
    
    inferred_type = 'unknown'
    
    try:
        prediction = model.predict(sample_data)
        
        if isinstance(prediction, np.ndarray):
            if prediction.ndim == 1:
                if len(prediction) == 1:
                    inferred_type = 'single_value'
                else:
                    inferred_type = 'array'
            else:
                inferred_type = 'matrix'
        elif isinstance(prediction, (int, float, np.number)):
            inferred_type = 'single_value'
        else:
            inferred_type = 'unknown'
    except Exception:
        try:
            prediction = model.predict_proba(sample_data)
            inferred_type = 'probabilities'
        except Exception:
            inferred_type = 'unknown'
    
    output_type = inferred_type
    return output_type


def create_predict_request_model():
    """Crea dinámicamente el modelo Pydantic para la request"""
    global PredictRequest
    
    if input_columns is None:
        raise ValueError("Las columnas de entrada no han sido inferidas")
    
    field_definitions = {}
    for col in input_columns:
        field_definitions[col] = (float, Field(..., description=f"Valor para {col}"))
    
    PredictRequest = create_model('PredictRequest', **field_definitions)
    return PredictRequest


def get_model_metadata() -> Dict[str, Any]:
    """Obtiene metadata del modelo para incluir en respuestas de error"""
    metadata = {
        "fecha": datetime.now().isoformat(),
        "version": MODEL_VERSION,
        "nombre": MODEL_NAME,
        "tipo": type(model).__name__ if model is not None else "unknown"
    }
    
    if model is not None:
        if hasattr(model, '__class__'):
            metadata["tipo"] = model.__class__.__name__
        
        if hasattr(model, 'get_params'):
            try:
                params = model.get_params()
                if 'steps' in params:
                    metadata["pipeline_steps"] = len(params['steps'])
            except Exception:
                pass
    
    metadata["entorno"] = ENVIRONMENT
    metadata["build_id"] = BUILD_ID
    
    return metadata


def handle_api_error(func: Callable) -> Callable:
    """Decorador que maneja errores y envuelve respuestas con metadata"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            if isinstance(result, dict):
                if "metadata" not in result:
                    result["metadata"] = {
                        "error": 0,
                        **get_model_metadata()
                    }
            else:
                result = {
                    "data": result,
                    "metadata": {
                        "error": 0,
                        **get_model_metadata()
                    }
                }
            
            return result
            
        except HTTPException as e:
            raise e
            
        except Exception as e:
            error_response = {
                "metadata": {
                    "error": 1,
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    **get_model_metadata()
                }
            }
            
            raise HTTPException(
                status_code=500,
                detail=error_response
            )
    
    return wrapper


def safe_api_call(func: Callable, input_data: Optional[Dict[str, Any]] = None, *args, **kwargs) -> Dict[str, Any]:
    """
    Función de soporte que ejecuta una llamada a API con manejo de errores.
    
    Si ocurre un error, retorna un objeto con:
    - predict_proba: valor por defecto en error (configurable)
    - predict: valor por defecto en error (configurable)
    - threshold: valor por defecto en error (si SHOW_THRESHOLD es true)
    - input: datos de entrada recibidos (si están disponibles)
    - metadata: contiene error: 1, error_message, error_type, y información del modelo
    
    Si no hay error, retorna la respuesta original con metadata que incluye error: 0
    """
    try:
        if input_data is not None:
            result = func(input_data, *args, **kwargs)
        else:
            result = func(*args, **kwargs)
        
        if isinstance(result, dict):
            if "metadata" not in result:
                result["metadata"] = {
                    "error": 0,
                    **get_model_metadata()
                }
            return result
        else:
            return {
                "data": result,
                "metadata": {
                    "error": 0,
                    **get_model_metadata()
                }
            }
            
    except HTTPException:
        raise
        
    except Exception as e:
        error_response = {
            OUTPUT_PREDICT_PROBA_KEY: ERROR_PREDICT_PROBA_VALUE,
            OUTPUT_PREDICT_KEY: ERROR_PREDICT_VALUE,
            "metadata": {
                "error": 1,
                "error_message": str(e),
                "error_type": type(e).__name__,
                **get_model_metadata()
            }
        }
        
        if SHOW_THRESHOLD:
            error_response[OUTPUT_THRESHOLD_KEY] = ERROR_THRESHOLD_VALUE
        
        if input_data is not None:
            error_response["input"] = input_data
        
        return error_response


@app.on_event("startup")
async def startup_event():
    """Inicializa el modelo y configura la API al arrancar"""
    global model, input_columns, output_type, PredictRequest
    
    try:
        framework = detect_framework()
        print(f"Framework detectado: {framework}")
        
        model = load_model()
        print(f"Modelo cargado desde {MODEL_PATH}")
        
        input_columns = infer_input_columns()
        print(f"Columnas de entrada inferidas: {input_columns}")
        
        output_type = infer_output_type()
        print(f"Tipo de salida inferido: {output_type}")
        
        PredictRequest = create_predict_request_model()
        print("Modelo de request creado dinámicamente")
        
    except Exception as e:
        print(f"Error durante la inicialización: {str(e)}")
        raise


@app.get("/")
def read_root():
    return {
        "message": "ML Inference API",
        "model_loaded": model is not None,
        "input_columns": input_columns,
        "output_type": output_type
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.get("/model-info")
def get_model_info():
    """Retorna información sobre el modelo"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    return {
        "input_columns": input_columns,
        "output_type": output_type,
        "model_type": type(model).__name__
    }


def _execute_prediction(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Ejecuta la predicción del modelo"""
    df_input = pd.DataFrame([input_data])
    df_input = df_input[input_columns]
    
    predict_proba_value = None
    predict_value = None
    
    try:
        proba = model.predict_proba(df_input)
        if isinstance(proba, np.ndarray):
            if proba.ndim == 2:
                predict_proba_value = float(proba[0][1])
            else:
                predict_proba_value = float(proba[0])
        else:
            predict_proba_value = float(proba)
    except (AttributeError, Exception):
        pass
    
    if predict_proba_value is not None:
        predict_value = 1 if predict_proba_value >= THRESHOLD else 0
    else:
        try:
            prediction_raw = model.predict(df_input)
            if isinstance(prediction_raw, np.ndarray):
                predict_value = int(prediction_raw[0])
            else:
                predict_value = int(prediction_raw)
            predict_proba_value = 1.0 if predict_value == 1 else 0.0
        except Exception:
            raise ValueError("El modelo no soporta predict_proba ni predict")
    
    result = {
        OUTPUT_PREDICT_PROBA_KEY: predict_proba_value,
        OUTPUT_PREDICT_KEY: predict_value,
        "input": input_data
    }
    
    if SHOW_THRESHOLD:
        result[OUTPUT_THRESHOLD_KEY] = THRESHOLD
    
    return result


@app.post("/predict")
def predict(request: Dict[str, Any]):
    """Endpoint de predicción que acepta JSON con las entradas del modelo"""
    input_data = {}
    
    if model is None:
        error_response = {
            OUTPUT_PREDICT_PROBA_KEY: ERROR_PREDICT_PROBA_VALUE,
            OUTPUT_PREDICT_KEY: ERROR_PREDICT_VALUE,
            "input": request,
            "metadata": {
                "error": 1,
                "error_message": "Modelo no cargado",
                "error_type": "ModelNotLoadedError",
                **get_model_metadata()
            }
        }
        if SHOW_THRESHOLD:
            error_response[OUTPUT_THRESHOLD_KEY] = ERROR_THRESHOLD_VALUE
        raise HTTPException(status_code=503, detail=error_response)
    
    if input_columns is None:
        error_response = {
            OUTPUT_PREDICT_PROBA_KEY: ERROR_PREDICT_PROBA_VALUE,
            OUTPUT_PREDICT_KEY: ERROR_PREDICT_VALUE,
            "input": request,
            "metadata": {
                "error": 1,
                "error_message": "Columnas de entrada no definidas",
                "error_type": "InputColumnsNotDefinedError",
                **get_model_metadata()
            }
        }
        if SHOW_THRESHOLD:
            error_response[OUTPUT_THRESHOLD_KEY] = ERROR_THRESHOLD_VALUE
        raise HTTPException(status_code=500, detail=error_response)
    
    try:
        for col in input_columns:
            if col not in request:
                error_response = {
                    OUTPUT_PREDICT_PROBA_KEY: ERROR_PREDICT_PROBA_VALUE,
                    OUTPUT_PREDICT_KEY: ERROR_PREDICT_VALUE,
                    "input": request,
                    "metadata": {
                        "error": 1,
                        "error_message": f"Falta la columna requerida: {col}",
                        "error_type": "MissingColumnError",
                        **get_model_metadata()
                    }
                }
                if SHOW_THRESHOLD:
                    error_response[OUTPUT_THRESHOLD_KEY] = ERROR_THRESHOLD_VALUE
                raise HTTPException(status_code=422, detail=error_response)
            input_data[col] = request[col]
        
        result = safe_api_call(_execute_prediction, input_data)
        
        if result.get("metadata", {}).get("error") == 1:
            if "input" not in result:
                result["input"] = input_data
            raise HTTPException(status_code=500, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        error_response = {
            OUTPUT_PREDICT_PROBA_KEY: ERROR_PREDICT_PROBA_VALUE,
            OUTPUT_PREDICT_KEY: ERROR_PREDICT_VALUE,
            "input": input_data if input_data else request,
            "metadata": {
                "error": 1,
                "error_message": str(e),
                "error_type": type(e).__name__,
                **get_model_metadata()
            }
        }
        if SHOW_THRESHOLD:
            error_response[OUTPUT_THRESHOLD_KEY] = ERROR_THRESHOLD_VALUE
        raise HTTPException(status_code=500, detail=error_response)
