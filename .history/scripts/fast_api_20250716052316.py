```python
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Query
from fastapi.security import APIKeyHeader
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uvicorn
import redis.asyncio as redis
import uuid
import time
import pandas as pd
import torch
import numpy as np
import asyncio
from functools import partial
import sqlite3
from contextlib import contextmanager
import hashlib
import secrets

# Database Setup (SQLite for Simplicity - Production: Use PostgreSQL)
DB_FILE = "evoai_hub.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (api_key TEXT PRIMARY KEY, email TEXT, tier TEXT, usage INTEGER DEFAULT 0, last_reset TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_FILE)
    try:
        yield conn
    finally:
        conn.close()

# Mock TE-AI (Replace with Real)
class MockTEAI:
    def evolve(self, domain: str, data: Dict, generations: int) -> Dict:
        num_solutions = 10
        solutions = [{"id": f"sol-{i}", "score": np.random.uniform(0.7, 0.95), "details": f"{domain} adapted"} for i in range(num_solutions)]
        metrics = {"mean_fitness": np.random.uniform(0.8, 0.9), "diversity_shannon": np.random.uniform(2.5, 4.0)}
        return {"evolved_solutions": sorted(solutions, key=lambda x: x["score"], reverse=True), "metrics": metrics}

teai_engine = MockTEAI()

app = FastAPI(
    title="EvoAI Hub API",
    description="Adaptive Intelligence Engine with Subdomain Accelerators",
    version="1.0"
)

# Redis for Rate Limiting
redis_client = redis.Redis.from_url("redis://localhost:6379/0", encoding="utf-8", decode_responses=True)

@app.on_event("startup")
async def startup():
    await FastAPILimiter.init(redis_client)

# API Key Validation from DB
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: Optional[str] = Depends(api_key_header)):
    if not api_key:
        raise HTTPException(401, "Missing API Key")
    
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT tier, usage FROM users WHERE api_key=?", (api_key,))
        user = c.fetchone()
        if not user:
            raise HTTPException(401, "Invalid API Key")
        
        tier, usage = user
        # Mock reset logic (e.g., monthly reset)
        # For demo, assume unlimited if tier='premium', else limit to 100
        if tier != 'premium' and usage >= 100:
            raise HTTPException(429, "Monthly usage limit reached")
    
    return api_key

# Log Usage
def log_usage(api_key: str, increment: int = 1):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET usage = usage + ? WHERE api_key=?", (increment, api_key))
        conn.commit()

# Schemas
class EvolveRequest(BaseModel):
    domain: str = Field(..., example="biotech")
    data: Dict = Field(..., example={"antigens": [[1.2, 3.4]]})
    generations: int = Field(5, ge=1, le=50)
    params: Optional[Dict] = Field(None, example={"stress_level": 0.7})

class EvolveResponse(BaseModel):
    job_id: str
    evolved_solutions: List[Dict]
    metrics: Dict
    logs: List[str]

# Core Endpoint with Usage Tracking
@app.post("/evolve", response_model=EvolveResponse, dependencies=[Depends(RateLimiter(times=100, seconds=3600))])
async def evolve(req: EvolveRequest, key: str = Depends(get_api_key)):
    start = time.time()
    job_id = str(uuid.uuid4())
    try:
        results = teai_engine.evolve(req.domain, req.data, req.generations)
        logs = [f"Evolution complete in {time.time() - start:.2f}s"]
        
        # Log usage after success
        log_usage(key)
        
        return {"job_id": job_id, **results, "logs": logs}
    except Exception as e:
        raise HTTPException(500, f"Evolution failed: {str(e)}")

# HTS Wrapper with Usage
@app.post("/hts/screen", dependencies=[Depends(RateLimiter(times=50, seconds=3600))])
async def hts_screen(
    omics_file: Optional[UploadFile] = File(None),
    disease: Optional[str] = Query("cancer"),
    num_targets: int = Query(50, ge=1, le=1000),
    key: str = Depends(get_api_key)
):
    if not omics_file:
        raise HTTPException(400, "Omics file required")
    
    content = await omics_file.read()
    df = pd.read_csv(io.BytesIO(content))
    req_data = {"omics": df.to_dict(orient="records")}

    # Internal Core Call (as "customer" - no extra usage log, since /evolve logs it)
    core_req = EvolveRequest(domain="biotech", data=req_data, generations=10)
    core_res = await evolve(core_req, key=key)  # Reuse endpoint logic
    
    ranked = core_res.evolved_solutions[:num_targets]
    return {
        "ranked_targets": ranked,
        "metrics": core_res.metrics,
        "powered_by": "EvoAI Engine - Customize at evoaihub.com/api",
        "visuals": {"heatmap": "mock_base64_png_data"}
    }

# Stats Endpoint
@app.get("/stats")
async def get_stats(domain: Optional[str] = None, key: str = Depends(get_api_key)):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT usage FROM users WHERE api_key=?", (key,))
        user_usage = c.fetchone()[0]
    
    # Mock global stats (in prod: Aggregate from logs DB)
    global_stats = {"avg_fitness": 0.85, "domain_specific": {domain: 0.88} if domain else {}}
    return {
        "user_runs": user_usage,
        "global_benchmarks": global_stats,
        "improvement_trends": [0.75, 0.82, 0.85]
    }

# User Registration (Mock - Prod: Email Verification)
@app.post("/auth/register")
async def register(email: str = Query(...)):
    api_key = secrets.token_hex(16)
    with get_db() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO users (api_key, email, tier, last_reset) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                  (api_key, email, 'free'))
        conn.commit()
    return {"api_key": api_key, "message": "Registered! Free tier: 100 reqs/month"}

# Run
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```