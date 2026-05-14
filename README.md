# EDIATH Autonomous AI System

Production-ready multi-agent AI with:
- Voice UI + backend
- LLM reasoning (GGUF)
- Agent orchestration
- Graceful shutdown
- No crashes/loops

## Quick Start
```bash
pip install -e .
python launcher.py --mode ui
```

## Modes
- `ui` (default): Full voice UI + backend
- `backend`: Headless orchestrator

## Optional Checks
```bash
python launcher.py --check-deps
python scripts/check_syntax.py
```
