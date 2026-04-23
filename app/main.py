import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from recommender import Recommender

load_dotenv()

app = FastAPI(title="Movie Recommendation System")
recommender = Recommender(os.getenv("DATA_PATH", "netflix_data.csv"))

STATIC_DIR = Path(__file__).parent.parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/recommend")
def recommend(
    query: str = Query(..., description="e.g. 'funny sci-fi movies with aliens'"),
    top_k: int = Query(10, ge=1, le=50),
):
    try:
        results = recommender.recommend(query, top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"query": query, "count": len(results), "results": results}
