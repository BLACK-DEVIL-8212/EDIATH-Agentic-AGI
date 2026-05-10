# EDIATH Autonomous AI System

Production-ready multi-agent AI with:
- Voice UI + backend
- LLM reasoning (GGUF)
- Agent orchestration
- Graceful shutdown
- No crashes/loops

## Quick Start
```
pip install -e .
python launcher.py --mode=ui
```

## Modes
- `ui` (default): Full voice UI + backend
- `backend`: Headless orchestrator (10min safe)
- `ui-only`: Kivy UI only

## Package
```
python -m ediath.cli --mode=backend  # DAG entry
pytest tests/  # Tests pass

