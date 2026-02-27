"""FastAPI server for SFUSD Zoning Dashboard."""
import json
import os
import secrets
import uuid
import logging
import traceback
import sys
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
from urllib.parse import unquote
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from data_loader import (
    load_zone_dict,
    get_zone_demographics,
    load_geojson,
    get_zone_color,
    load_solution_result,
    compute_percentile_ranks,
    get_category_percentiles,
    get_school_locations,
    get_all_metrics_stats,
    get_pareto_solutions,
    filter_and_centroid,
    suggest_relaxation,
)
from LLM.exploration.zoning_agent import ZoningAgent
from Zone_Generation.Config.metrics_config import (
    ALL_METRICS, CATEGORIES, CATEGORY_DESCRIPTIONS,
    ETHNICITY_DISPLAY_LABELS, get_chart_hints,
)
from Zone_Generation.Config.Constants import PROGRAM_NAMES, AREA_ETHNICITIES

# Add project root to path for LLM imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the solutions CSV
DEFAULT_CSV_PATH = "/share/data/school_choice/local_runs/kumar_website_test/new_benchmarks_test/summary.csv"

# Session storage for ZoningAgent instances
agent_sessions: dict[str, ZoningAgent] = {}

# Admin auth
ADMIN_CONFIG_PATH = Path(__file__).parent / "admin_config.json"
_admin_password = "sfusd-admin-2026"
if ADMIN_CONFIG_PATH.exists():
    with open(ADMIN_CONFIG_PATH) as f:
        _admin_password = json.load(f).get("password", _admin_password)
_admin_tokens: set[str] = set()

app = FastAPI(title="SFUSD Zoning Dashboard API")

# Pre-loaded agent for faster first response
_preloaded_agent: Optional[ZoningAgent] = None


@app.on_event("startup")
def startup_event():
    """Pre-load the agent on startup to avoid first-request delay."""
    global _preloaded_agent
    logger.info("Pre-loading ZoningAgent on startup...")
    try:
        _preloaded_agent = ZoningAgent(DEFAULT_CSV_PATH)
        logger.info("ZoningAgent pre-loaded successfully")
    except Exception as e:
        logger.error(f"Failed to pre-load agent: {e}")
        logger.error(traceback.format_exc())

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
    session_id: Optional[str] = None
    current_solution_index: Optional[int] = None
    saved_solutions: Optional[list] = None  # [{index, label, pros, cons, key_metrics}]


class AdminAuthRequest(BaseModel):
    password: str


class AdminFilterRequest(BaseModel):
    bounds: dict  # {metric_column: {min_bound, max_bound}}


@app.get("/")
async def root():
    """Serve the frontend."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico")
async def favicon():
    """Serve the favicon."""
    return FileResponse(FRONTEND_DIR / "favicon.ico")


@app.get("/api/health")
def health_check():
    """Health check endpoint to verify agent status."""
    return {
        "status": "ok",
        "agent_preloaded": _preloaded_agent is not None,
        "active_sessions": len(agent_sessions),
    }


@app.get("/api/config")
async def get_config():
    """Serve frontend configuration including PostHog API key."""
    return {"posthog_api_key": os.getenv("POSTHOG_API_KEY", "")}


@app.get("/api/solution/{path:path}")
async def get_solution(path: str):
    """
    Get zone assignments, demographics, and metrics for a specific solution.

    Returns:
    - zones: dict mapping blockgroup_id to zone_id
    - zone_data: dict mapping zone_id to comprehensive zone statistics (demographics, programs, quality metrics)
    - metrics: dict of solution-level metrics (FRL, ethnicities, distances, etc.)
    - colors: dict mapping zone_id to hex color
    - status: optimization status
    """
    try:
        decoded_path = unquote(path)
        bg_zone_dict = load_zone_dict(decoded_path)
        zone_data = get_zone_demographics(decoded_path)

        # Load full result for metrics
        result = load_solution_result(decoded_path)

        # Build zone index map (sorted zone IDs -> 1-indexed integers)
        zone_ids = sorted(set(bg_zone_dict.values()))
        zone_index_map = {zone_id: idx + 1 for idx, zone_id in enumerate(zone_ids)}

        # Build colors map (using original zone IDs for lookup)
        colors = {zone_id: get_zone_color(zone_id) for zone_id in zone_ids}

        # Convert keys to strings for JSON
        zones = {str(k): v for k, v in bg_zone_dict.items()}
        zone_data_json = {str(k): v for k, v in zone_data.items()}
        colors = {str(k): v for k, v in colors.items()}
        zone_index_map_json = {str(k): v for k, v in zone_index_map.items()}

        solution_metrics = result.get("metrics", {})
        pct_ranks = compute_percentile_ranks(solution_metrics, solution_path=decoded_path)
        return {
            "zones": zones,
            "zone_data": zone_data_json,
            "metrics": solution_metrics,
            "percentile_ranks": pct_ranks,
            "category_percentiles": get_category_percentiles(
                solution_path=decoded_path, percentile_ranks=pct_ranks
            ),
            "status": result.get("status", "UNKNOWN"),

            "boundary_cost": result.get("boundary_cost"),
            "total_wall_time": result.get("total_wall_time"),
            "colors": colors,
            "zone_index_map": zone_index_map_json,
            "path": decoded_path,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error loading solution: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/geojson")
async def get_geojson_data():
    """Get the GeoJSON data for SF blockgroups."""
    try:
        geojson = load_geojson()
        return JSONResponse(content=geojson)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/schools")
async def get_schools():
    """Get all school locations with lat/lon coordinates."""
    try:
        schools = get_school_locations()
        return {"schools": schools}
    except Exception as e:
        logger.error(f"Error loading school locations: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def get_or_create_agent(session_id: Optional[str]) -> tuple[str, ZoningAgent]:
    """Get existing agent for session or create a new one."""
    global _preloaded_agent

    if session_id and session_id in agent_sessions:
        logger.info(f"Reusing existing agent for session {session_id}")
        return session_id, agent_sessions[session_id]

    # Create new session
    new_session_id = str(uuid.uuid4())

    # Use pre-loaded agent for first session if available
    if _preloaded_agent is not None:
        logger.info(f"Using pre-loaded agent for session {new_session_id}")
        agent = _preloaded_agent
        _preloaded_agent = None  # Only use once
    else:
        logger.info(f"Creating new agent for session {new_session_id}...")
        agent = ZoningAgent(DEFAULT_CSV_PATH)
        logger.info(f"Agent created for session {new_session_id}")

    agent_sessions[new_session_id] = agent
    return new_session_id, agent


@app.post("/api/chat")
def chat(request: ChatRequest):
    """Chat with the ZoningAgent (sync endpoint for blocking LLM calls)."""
    try:
        logger.info(f"Chat request: {request.message[:100]}...")
        session_id, agent = get_or_create_agent(request.session_id)

        # Build solution context from request
        solution_context = None
        if request.saved_solutions or request.current_solution_index is not None:
            solution_context = {
                "current_solution_index": request.current_solution_index,
                "saved_solutions": request.saved_solutions or [],
            }

        logger.info("Calling agent.chat...")
        result = agent.chat(request.message, solution_context=solution_context)
        logger.info(f"Agent response type: {result.get('response_type')}")
        logger.info(f"Agent text length: {len(result.get('text', ''))}")

        return {
            "text": result["text"],
            "response_type": result["response_type"],
            "clusters": result.get("clusters"),
            "solution_path": result.get("solution_path"),
            "description": result.get("description", ""),
            "session_id": session_id,
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Metrics config endpoint (single source of truth for frontend)
# ============================================================================

_chart_hints = get_chart_hints()


@app.get("/api/metrics-config")
async def get_metrics_config():
    """Serve centralized metrics configuration for frontend consumption."""
    categories = {
        key: {"name": name, "description": CATEGORY_DESCRIPTIONS.get(key, "")}
        for key, name in CATEGORIES.items()
    }

    metrics = []
    for m in ALL_METRICS:
        entry = {
            "column": m.column,
            "display_name": m.display_name,
            "description": m.description,
            "category": m.category,
            "direction": m.direction,
            "is_core": m.is_core,
            "short_name": m.short_name or m.display_name[:4],
            "chart": _chart_hints.get(m.column, {"type": "none"}),
        }
        metrics.append(entry)

    ethnicities_display = [
        {"key": key, "label": ETHNICITY_DISPLAY_LABELS.get(key, key.replace("Ethnicity_", "").replace("_", " "))}
        for key in AREA_ETHNICITIES
    ]

    return {
        "categories": categories,
        "metrics": metrics,
        "programs": PROGRAM_NAMES,
        "ethnicities": {"display": ethnicities_display},
    }


# ============================================================================
# Admin console endpoints
# ============================================================================

def _verify_admin(authorization: Optional[str]) -> None:
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token not in _admin_tokens:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/admin")
async def admin_page():
    """Serve the admin console."""
    return FileResponse(FRONTEND_DIR / "admin.html")


@app.post("/api/admin/auth")
async def admin_auth(request: AdminAuthRequest):
    if request.password != _admin_password:
        raise HTTPException(status_code=401, detail="Invalid password")
    token = secrets.token_urlsafe(32)
    _admin_tokens.add(token)
    return {"token": token}


@app.get("/api/admin/solution-space")
async def admin_solution_space(authorization: Optional[str] = Header(None)):
    _verify_admin(authorization)
    stats = get_all_metrics_stats()
    pareto = get_pareto_solutions()
    return {
        "metrics": stats,
        "total_pareto": len(pareto),
        "categories": {
            k: v for k, v in {
                "diversity": "Demographics & Economic Balance",
                "distance": "Geographic Access & Proximity",
                "programs": "Educational Program Availability",
                "quality": "School Quality Indicators",
                "structure": "Zone Structure & Shape",
            }.items()
        },
    }


@app.post("/api/admin/filter")
async def admin_filter(request: AdminFilterRequest, authorization: Optional[str] = Header(None)):
    _verify_admin(authorization)
    result = filter_and_centroid(request.bounds)
    return result


@app.post("/api/admin/suggest-relaxation")
async def admin_suggest_relaxation(request: AdminFilterRequest, authorization: Optional[str] = Header(None)):
    _verify_admin(authorization)
    suggestions = suggest_relaxation(request.bounds)
    return {"suggestions": suggestions}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
