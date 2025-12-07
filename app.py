from fastapi import FastAPI

app = FastAPI(title="Hola Mundo API")


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a friendly greeting."""
    return {"message": "hola mundo"}
