"""FastAPI server for SFUSD Zoning Dashboard."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from pydantic import BaseModel
from urllib.parse import unquote

from data_loader import (
    get_clusters,
    load_zone_dict,
    get_zone_demographics,
    load_geojson,
    get_zone_color,
)

app = FastAPI(title="SFUSD Zoning Dashboard API")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class ChatRequest(BaseModel):
    message: str
    cluster_label: str = ""


@app.get("/")
async def root():
    """Serve the frontend."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/clusters")
async def get_solution_clusters():
    """Get clustered solutions with labels and representatives."""
    try:
        clusters = get_clusters()
        return {"clusters": clusters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/solution/{path:path}")
async def get_solution(path: str):
    """
    Get zone assignments and demographics for a specific solution.

    Returns:
    - zones: dict mapping blockgroup_id to zone_id
    - demographics: dict mapping zone_id to demographic stats
    - colors: dict mapping zone_id to hex color
    """
    try:
        decoded_path = unquote(path)
        bg_zone_dict = load_zone_dict(decoded_path)
        demographics = get_zone_demographics(bg_zone_dict)

        # Build colors map
        zone_ids = set(bg_zone_dict.values())
        colors = {zone_id: get_zone_color(zone_id) for zone_id in zone_ids}

        # Convert keys to strings for JSON
        zones = {str(k): v for k, v in bg_zone_dict.items()}
        demographics = {str(k): v for k, v in demographics.items()}
        colors = {str(k): v for k, v in colors.items()}

        return {
            "zones": zones,
            "demographics": demographics,
            "colors": colors,
            "path": decoded_path,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Solution not found: {path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/geojson")
async def get_geojson_data():
    """Get the GeoJSON data for SF blockgroups."""
    try:
        geojson = load_geojson()
        return JSONResponse(content=geojson)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Placeholder chat endpoint."""
    label_info = f" Currently viewing: {request.cluster_label}" if request.cluster_label else ""
    return {
        "response": f"Agent connection coming soon.{label_info}",
        "agent_connected": False,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
