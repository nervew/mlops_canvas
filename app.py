from datetime import date
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "hola mundo"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


class FechaNacimiento(BaseModel):
    fecha_nacimiento: str

    @field_validator("fecha_nacimiento")
    @classmethod
    def validar_formato(cls, v):
        try:
            fecha = date.fromisoformat(v)
            if fecha > date.today():
                raise ValueError("La fecha de nacimiento no puede ser futura")
            return v
        except ValueError as e:
            if "invalid date format" in str(e).lower() or "invalid isoformat" in str(e).lower():
                raise ValueError("Formato de fecha inválido. Use YYYY-MM-DD")
            raise


@app.post("/edad")
def calcular_edad(fecha: FechaNacimiento):
    fecha_nac = date.fromisoformat(fecha.fecha_nacimiento)
    hoy = date.today()
    edad = hoy.year - fecha_nac.year - ((hoy.month, hoy.day) < (fecha_nac.month, fecha_nac.day))
    return {"edad": edad}

