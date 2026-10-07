import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from goat_weight import make_router

app = FastAPI(title="GoatVision weight service")

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(make_router())


@app.get("/api/weight/health")
def health():
    return {"status": "ok"}
