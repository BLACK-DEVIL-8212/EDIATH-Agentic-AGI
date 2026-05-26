from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pathlib import Path
import uvicorn

from core.system.orchestrator import EDIATHOrchestrator, OrchestratorConfig

app = FastAPI(title="EDIATH Web UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent

# Serve static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Single orchestrator instance for UI
_orchestrator: EDIATHOrchestrator | None = None
_orch_lock = asyncio.Lock()


async def get_orchestrator() -> EDIATHOrchestrator:
    global _orchestrator
    async with _orch_lock:
        if _orchestrator is None:
            cfg_path = Path("config/orchestrator_config.yaml")
            if cfg_path.exists():
                cfg = OrchestratorConfig.from_yaml(cfg_path)
            else:
                cfg = OrchestratorConfig()
            _orchestrator = EDIATHOrchestrator(cfg)
        return _orchestrator


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.post("/initialize")
async def initialize():
    orch = await get_orchestrator()
    try:
        ok = await orch.initialize()
        return {"ok": bool(ok)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run")
async def run():
    orch = await get_orchestrator()
    # run in background
    loop = asyncio.get_running_loop()
    loop.create_task(orch.run())
    return {"started": True}


@app.post("/shutdown")
async def shutdown():
    global _orchestrator
    if _orchestrator is None:
        return {"ok": True}
    try:
        await _orchestrator.shutdown(graceful=True)
        _orchestrator = None
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status():
    orch = await get_orchestrator()
    try:
        return orch.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
async def agents():
    orch = await get_orchestrator()
    try:
        return {"agents": list(orch.get_all_agents().keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute")
async def execute(command: dict):
    orch = await get_orchestrator()
    try:
        system_ctrl = orch.components.get("system_controller")
        if system_ctrl and hasattr(system_ctrl, "execute"):
            res = await orch._safe_call(system_ctrl, "execute", command.get("text", ""))
            return {"result": res}
        return {"error": "No system_controller available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("webui.app:app", host="0.0.0.0", port=8000, reload=False)
