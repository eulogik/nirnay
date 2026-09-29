"""Jev-compatible server: POST /v1/systemone.

Wire format (verified from jevbench/adapters/typesafe.py):
  Request:  {"state": ..., "model": ..., "questions": {"decision": {...}}}
  Response: {"model": ..., "answers": {"decision": {...}}, "usage": {...}}

noul → answers.decision.noul in [0,1]
choice → answers.decision.choice + .probabilities (keys = options)
score → answers.decision.probabilities (keys = level index strings)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import NirnayAgent

_agent: Optional[NirnayAgent] = None


class Question(BaseModel):
    type: str
    instructions: Any
    criteria: Optional[Any] = None


class SystemOneRequest(BaseModel):
    state: Any
    model: Optional[str] = "nirnay-1"
    questions: dict[str, Question] = Field(default_factory=dict)


def get_agent() -> NirnayAgent:
    global _agent
    if _agent is None:
        path = os.environ.get("NIRNAY_MODEL", "convaiinnovations/laya")
        device = os.environ.get("NIRNAY_DEVICE")  # None → auto
        temps = os.environ.get("NIRNAY_TEMPS")
        checkpoint = os.environ.get("NIRNAY_CHECKPOINT")
        lora_rank = int(os.environ.get("NIRNAY_LORA_RANK", "8"))
        _agent = NirnayAgent(
            path,
            device=device,
            temps_path=temps or None,
            checkpoint_path=checkpoint or None,
            lora_rank=lora_rank,
        )

    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_agent()  # warm load
    yield


app = FastAPI(title="NIRNAY-1", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    a = get_agent()
    return {"ok": True, **a.report()}


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest) -> dict:
    if not req.questions:
        raise HTTPException(status_code=400, detail="questions required")
    agent = get_agent()
    qdefs = {k: q.model_dump(exclude_none=True) for k, q in req.questions.items()}
    try:
        out = agent.system_one(req.state, qdefs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    # Ensure model field reflects request model alias when provided
    if req.model:
        out = {**out, "model": req.model}
    return out


def main() -> None:
    import uvicorn

    host = os.environ.get("NIRNAY_HOST", "127.0.0.1")
    port = int(os.environ.get("NIRNAY_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
