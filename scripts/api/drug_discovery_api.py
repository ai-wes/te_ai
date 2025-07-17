"""
TE-AI Drug Discovery API Service
================================

FastAPI service for drug target narrowing using the full TE-AI framework.
Provides RESTful endpoints for omics data upload, target evaluation,
and results retrieval.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import torch
import json
import uuid
import time
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import traceback

from scripts.domains.drug_discovery import (
    DrugDiscoveryGerminalCenter,
    OmicsToAntigenConverter,
    DrugTargetEvaluator,
    DrugTargetAntigen
)
from scripts.domains.drug_discovery.omics_to_antigen_converter import OmicsData
from scripts.core.anitgen import generate_realistic_antigen
from scripts.config import cfg


# Pydantic models for API
class EvaluationRequest(BaseModel):
    """Request model for drug target evaluation"""
    job_id: Optional[str] = Field(None, description="Optional job ID for tracking")
    target_proteins: Optional[List[str]] = Field(None, description="Specific proteins to evaluate")
    disease_focus: Optional[str] = Field(None, description="Disease context")
    generations: int = Field(20, description="Evolution generations per target")
    population_size: Optional[int] = Field(None, description="Override default population size")
    enable_quantum: bool = Field(True, description="Enable quantum processing")
    enable_dreams: bool = Field(True, description="Enable dream consolidation")
    stress_test: bool = Field(True, description="Test mutation resistance")
    parallel_evaluation: bool = Field(True, description="Evaluate targets in parallel")


class EvaluationResponse(BaseModel):
    """Response model for evaluation request"""
    job_id: str
    status: str
    message: str
    estimated_time: Optional[float] = None


class TargetScore(BaseModel):
    """Drug target score summary"""
    target_id: str
    overall_score: float
    components: Dict[str, float]
    quantum_coherence: float
    evolutionary_potential: float
    rank: int
    recommendation: str


class JobStatus(BaseModel):
    """Job status information"""
    job_id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    progress: float  # 0.0 to 1.0
    current_target: Optional[str] = None
    targets_completed: int = 0
    total_targets: int = 0
    elapsed_time: float = 0.0
    estimated_remaining: Optional[float] = None
    error: Optional[str] = None


class ResultsResponse(BaseModel):
    """Complete results for a job"""
    job_id: str
    status: str
    completed_at: str
    evaluation_time: float
    summary: Dict[str, Any]
    ranked_targets: List[TargetScore]
    detailed_results: Optional[Dict[str, Any]] = None


# Initialize FastAPI app
app = FastAPI(
    title="TE-AI Drug Discovery API",
    description="Transposable Element AI for Drug Target Narrowing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global job storage (in production, use Redis or database)
jobs_db = {}
results_db = {}

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=4)


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    # Ensure CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available, using CPU")
        
    # Create directories
    os.makedirs("api_uploads", exist_ok=True)
    os.makedirs("api_results", exist_ok=True)


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "TE-AI Drug Discovery",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "evaluate": "/evaluate",
            "status": "/status/{job_id}",
            "results": "/results/{job_id}",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "active_jobs": len([j for j in jobs_db.values() if j["status"] == "running"]),
        "completed_jobs": len(results_db)
    }


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_targets(
    background_tasks: BackgroundTasks,
    request: EvaluationRequest,
    omics_file: Optional[UploadFile] = File(None, description="Omics data CSV"),
    structure_file: Optional[UploadFile] = File(None, description="Structure data JSON")
):
    """
    Evaluate drug targets from omics data.
    
    Accepts CSV file with omics data and optional structure information.
    Returns job ID for tracking evaluation progress.
    """
    # Generate job ID
    job_id = request.job_id or str(uuid.uuid4())
    
    # Validate inputs
    if not omics_file and not request.target_proteins:
        raise HTTPException(
            status_code=400,
            detail="Either omics_file or target_proteins must be provided"
        )
        
    # Save uploaded files
    omics_path = None
    structure_path = None
    
    try:
        if omics_file:
            omics_path = f"api_uploads/{job_id}_omics.csv"
            content = await omics_file.read()
            with open(omics_path, 'wb') as f:
                f.write(content)
                
        if structure_file:
            structure_path = f"api_uploads/{job_id}_structure.json"
            content = await structure_file.read()
            with open(structure_path, 'wb') as f:
                f.write(content)
                
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File upload failed: {str(e)}")
        
    # Initialize job
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "created_at": datetime.now().isoformat(),
        "config": request.dict(),
        "omics_path": omics_path,
        "structure_path": structure_path
    }
    
    # Start background evaluation
    background_tasks.add_task(
        run_evaluation,
        job_id,
        omics_path,
        structure_path,
        request
    )
    
    # Estimate time (rough estimate: 30 seconds per target)
    estimated_targets = len(request.target_proteins) if request.target_proteins else 10
    estimated_time = estimated_targets * 30.0
    
    return EvaluationResponse(
        job_id=job_id,
        status="accepted",
        message="Evaluation started",
        estimated_time=estimated_time
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of evaluation job"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs_db[job_id]
    
    # Calculate elapsed time
    created_at = datetime.fromisoformat(job["created_at"])
    elapsed = (datetime.now() - created_at).total_seconds()
    
    # Estimate remaining time
    remaining = None
    if job["progress"] > 0 and job["progress"] < 1.0:
        total_estimated = elapsed / job["progress"]
        remaining = total_estimated - elapsed
        
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        current_target=job.get("current_target"),
        targets_completed=job.get("targets_completed", 0),
        total_targets=job.get("total_targets", 0),
        elapsed_time=elapsed,
        estimated_remaining=remaining,
        error=job.get("error")
    )


@app.get("/results/{job_id}", response_model=ResultsResponse)
async def get_results(
    job_id: str,
    detailed: bool = Query(False, description="Include detailed results")
):
    """Get evaluation results"""
    if job_id not in results_db:
        if job_id in jobs_db:
            return JSONResponse(
                status_code=202,
                content={"message": "Job still running", "status": jobs_db[job_id]["status"]}
            )
        else:
            raise HTTPException(status_code=404, detail="Job not found")
            
    results = results_db[job_id]
    
    # Format response
    response = ResultsResponse(
        job_id=job_id,
        status="completed",
        completed_at=results["completed_at"],
        evaluation_time=results["evaluation_time"],
        summary=results["summary"],
        ranked_targets=results["ranked_targets"]
    )
    
    if detailed:
        response.detailed_results = results.get("detailed_results")
        
    return response


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete job and associated data"""
    if job_id not in jobs_db and job_id not in results_db:
        raise HTTPException(status_code=404, detail="Job not found")
        
    # Clean up files
    for filename in os.listdir("api_uploads"):
        if filename.startswith(job_id):
            os.remove(os.path.join("api_uploads", filename))
            
    for filename in os.listdir("api_results"):
        if filename.startswith(job_id):
            os.remove(os.path.join("api_results", filename))
            
    # Remove from databases
    jobs_db.pop(job_id, None)
    results_db.pop(job_id, None)
    
    return {"message": "Job deleted successfully"}


def run_evaluation(
    job_id: str,
    omics_path: Optional[str],
    structure_path: Optional[str],
    request: EvaluationRequest
):
    """Run the actual evaluation in background"""
    try:
        # Update job status
        jobs_db[job_id]["status"] = "running"
        jobs_db[job_id]["started_at"] = datetime.now().isoformat()
        
        # Load data and create antigens
        converter = OmicsToAntigenConverter()
        
        if omics_path:
            # Load omics data
            omics_df = pd.read_csv(omics_path)
            
            # Create OmicsData object
            omics_data = OmicsData(
                gene_expression=omics_df if 'expression' in omics_df.columns else None,
                protein_abundance=omics_df if 'abundance' in omics_df.columns else None
            )
            
            # Load structure data if provided
            if structure_path:
                with open(structure_path, 'r') as f:
                    structure_data = json.load(f)
                omics_data.structural_data = structure_data
                
            # Convert to antigens
            antigens = converter.convert_omics_to_antigens(
                omics_data,
                target_proteins=request.target_proteins,
                disease_focus=request.disease_focus
            )
        else:
            # Create mock antigens from protein list
            antigens = []
            for protein_id in request.target_proteins:
                # Create mock omics data for protein
                mock_omics = OmicsData()
                antigen = converter._create_antigen_from_protein(
                    protein_id, mock_omics, request.disease_focus
                )
                if antigen:
                    antigens.append(antigen)
                    
        # Update job with target count
        jobs_db[job_id]["total_targets"] = len(antigens)
        
        # Initialize germinal center
        gc = DrugDiscoveryGerminalCenter(
            population_size=request.population_size,
            enable_quantum_dreams=request.enable_dreams
        )
        
        # Initialize evaluator
        evaluator = DrugTargetEvaluator(
            germinal_center=gc,
            quantum_evaluation=request.enable_quantum,
            dream_analysis=request.enable_dreams
        )
        
        # Progress tracking
        def update_progress(target_idx, target_id):
            jobs_db[job_id]["targets_completed"] = target_idx
            jobs_db[job_id]["current_target"] = target_id
            jobs_db[job_id]["progress"] = target_idx / len(antigens)
            
        # Evaluate targets
        start_time = time.time()
        
        # Run evaluation with progress updates
        scores = {}
        for i, target in enumerate(antigens):
            target_id = target.protein_structure.pdb_id or f"target_{i}"
            update_progress(i, target_id)
            
            # Single target evaluation
            score = evaluator._evaluate_single_target(
                target,
                generations=request.generations,
                stress_test=request.stress_test
            )
            scores[target_id] = score
            
        # Final progress update
        update_progress(len(antigens), "complete")
        
        evaluation_time = time.time() - start_time
        
        # Generate report
        report = gc.generate_druggability_report(
            {tid: {
                'target': antigens[i],
                'druggability_score': score.overall_score,
                'top_binders': [
                    {
                        'cell_id': prof['cell_id'],
                        'affinity': prof['affinity'],
                        'gene_signature': prof['gene_signature']
                    }
                    for prof in score.binding_profiles[:3]
                ],
                'mutation_resistance': {},
                'population_diversity': {}
            } for i, (tid, score) in enumerate(scores.items())}
        )
        
        # Format results
        ranked_targets = []
        for rank, (target_id, score) in enumerate(
            sorted(scores.items(), key=lambda x: x[1].overall_score, reverse=True),
            start=1
        ):
            recommendation = "Highly Recommended" if score.overall_score > 0.8 else \
                           "Recommended" if score.overall_score > 0.6 else \
                           "Moderate Potential" if score.overall_score > 0.4 else \
                           "Low Priority"
                           
            ranked_targets.append(TargetScore(
                target_id=target_id,
                overall_score=score.overall_score,
                components=score.components,
                quantum_coherence=score.quantum_coherence,
                evolutionary_potential=score.evolutionary_potential,
                rank=rank,
                recommendation=recommendation
            ))
            
        # Store results
        results_db[job_id] = {
            "job_id": job_id,
            "completed_at": datetime.now().isoformat(),
            "evaluation_time": evaluation_time,
            "summary": report["summary"],
            "ranked_targets": [t.dict() for t in ranked_targets],
            "detailed_results": report["detailed_results"]
        }
        
        # Save results to file
        results_path = f"api_results/{job_id}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_db[job_id], f, indent=2)
            
        # Update job status
        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["progress"] = 1.0
        
    except Exception as e:
        # Handle errors
        error_msg = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()
        
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error"] = error_msg
        
        # Store partial results if any
        results_db[job_id] = {
            "job_id": job_id,
            "status": "failed",
            "error": error_msg,
            "completed_at": datetime.now().isoformat()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)