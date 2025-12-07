from fastapi import FastAPI

app = FastAPI(title="mlopstest-api")


@app.get("/")
async def root() -> dict[str, str]:
    """Return a simple greeting for health checks."""
    return {"mensaje": "hola mundo"}


@app.get("/healthz")
async def healthcheck() -> dict[str, str]:
    """Basic health endpoint for container probes."""
    return {"status": "ok"}
