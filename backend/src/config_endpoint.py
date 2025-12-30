from fastapi import FastAPI
from backend.src.config import CONFIG_SOURCES

app = FastAPI()

@app.get("/config")
def get_config():
    """
    Returns all config keys, their values, and source (env or default).
    """
    return CONFIG_SOURCES
