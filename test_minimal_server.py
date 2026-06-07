"""Minimal FastAPI server for testing endpoints locally."""
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="SentiYelp Test Server", version="0.1.0")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "test"}

@app.get("/")
async def root():
    return {"message": "SentiYelp Test Server Running", "endpoints": ["/health", "/predict"]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
