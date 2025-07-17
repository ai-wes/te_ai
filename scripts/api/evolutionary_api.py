"""
TE-AI Evolutionary Algorithm API Service
========================================

REST API for running drug discovery and other evolutionary tasks
using the Transposable Element AI system.
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import json
import uuid
from pathlib import Path
import torch
import numpy as np

# Import TE-AI components
from scripts.config import Config
from scripts.domains.drug_discovery.drug_discovery_germinal_center import DrugDiscoveryGerminalCenter
from scripts.domains.drug_discovery.drug_target_evaluator import DrugTargetEvaluator
from scripts.domains.drug_discovery.tcga_adapter import TCGAAdapter
from scripts.domains.drug_discovery.omics_to_antigen_converter import OmicsToAntigenConverter, OmicsData
from scripts.core.utils.detailed_logger import get_logger

logger = get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="TE-AI Evolutionary Algorithm API",
    description="API for running evolutionary drug discovery and optimization tasks",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for running jobs
running_jobs: Dict[str, Dict] = {}
job_results: Dict[str, Dict] = {}


class EvolutionRequest(BaseModel):
    """Request model for starting an evolution job"""
    task_type: str = Field(default="drug_discovery", description="Type of evolution task")
    generations: int = Field(default=50, ge=1, le=1000, description="Number of generations to evolve")
    population_size: Optional[int] = Field(default=64, ge=10, le=500, description="Initial population size")
    target_proteins: Optional[List[str]] = Field(default=None, description="Specific proteins to target")
    disease_focus: Optional[str] = Field(default=None, description="Disease context for optimization")
    tcga_samples: Optional[List[str]] = Field(default=None, description="TCGA sample IDs to use")
    omics_data: Optional[Dict[str, Any]] = Field(default=None, description="Custom omics data")
    enable_quantum_dreams: bool = Field(default=True, description="Enable quantum dream consolidation")
    enable_drug_genes: bool = Field(default=True, description="Enable specialized drug discovery genes")


class EvolutionResponse(BaseModel):
    """Response model for evolution job submission"""
    job_id: str
    status: str
    message: str
    created_at: datetime


class JobStatus(BaseModel):
    """Model for job status information"""
    job_id: str
    status: str
    current_generation: int
    total_generations: int
    population_size: int
    best_fitness: float
    diversity_score: float
    elapsed_time: float
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DrugTargetResult(BaseModel):
    """Model for drug target discovery results"""
    protein_id: str
    druggability_score: float
    binding_pockets: List[Dict[str, Any]]
    selectivity_score: float
    disease_relevance: Optional[float] = None
    evolved_binders: List[Dict[str, Any]]


async def run_drug_discovery_evolution(
    job_id: str,
    request: EvolutionRequest
):
    """Background task to run drug discovery evolution"""
    try:
        # Update job status
        running_jobs[job_id]["status"] = "initializing"
        running_jobs[job_id]["start_time"] = datetime.now()
        
        # Initialize components
        logger.info(f"Starting drug discovery job {job_id}")
        
        # Create germinal center
        germinal_center = DrugDiscoveryGerminalCenter(
            population_size=request.population_size,
            enable_quantum_dreams=request.enable_quantum_dreams,
            enable_drug_genes=request.enable_drug_genes
        )
        
        # Prepare target antigens
        if request.tcga_samples:
            # Use TCGA data
            tcga_adapter = TCGAAdapter()
            converter = OmicsToAntigenConverter()
            
            # Convert TCGA samples
            samples = tcga_adapter.load_samples(request.tcga_samples)
            target_ids = tcga_adapter.identify_drug_targets_from_tcga(
                samples, 
                top_k=20
            )
            drug_targets = tcga_adapter.convert_tcga_to_antigens(
                target_ids, 
                samples
            )
        elif request.omics_data:
            # Use custom omics data
            converter = OmicsToAntigenConverter()
            omics = OmicsData(**request.omics_data)
            drug_targets = converter.convert_omics_to_antigens(
                omics,
                target_proteins=request.target_proteins,
                disease_focus=request.disease_focus
            )
        else:
            # Generate mock targets for demo
            from scripts.domains.drug_discovery.drug_target_antigen import DrugTargetAntigen, ProteinStructure, BindingPocket
            drug_targets = []
            for i in range(10):
                structure = ProteinStructure(
                    sequence="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL",
                    coordinates=np.random.randn(300, 3),
                    secondary_structure="CCCHHHHHHHHHCCCCEEEEEEECCCHHHHHHCCCEEEEEEE"
                )
                pocket = BindingPocket(
                    pocket_id=f"pocket_{i}",
                    residue_indices=list(range(10, 30)),
                    volume=350.0,
                    hydrophobicity=0.4,
                    electrostatic_potential=0.1,
                    druggability_score=0.75
                )
                target = DrugTargetAntigen(
                    protein_structure=structure,
                    binding_pockets=[pocket],
                    disease_association=request.disease_focus
                )
                drug_targets.append(target)
        
        running_jobs[job_id]["status"] = "evolving"
        running_jobs[job_id]["total_targets"] = len(drug_targets)
        
        # Evolution loop
        evolution_history = []
        for generation in range(request.generations):
            # Check if job was cancelled
            if running_jobs[job_id].get("cancelled", False):
                break
                
            # Sample targets for this generation
            import random
            target_batch = random.sample(
                drug_targets, 
                k=min(Config().batch_size, len(drug_targets))
            )
            
            # Convert to graphs
            graph_batch = [target.to_graph() for target in target_batch]
            
            # Evolve one generation
            stats = germinal_center.evolve_generation(graph_batch)
            
            # Update job progress
            if stats:
                running_jobs[job_id].update({
                    "current_generation": generation + 1,
                    "population_size": stats.get("population_size", 0),
                    "best_fitness": stats.get("best_fitness", 0.0),
                    "diversity_score": stats.get("diversity", 0.0),
                    "metrics": stats
                })
                evolution_history.append(stats)
            
            # Yield control to allow other tasks
            await asyncio.sleep(0.1)
        
        # Final evaluation
        running_jobs[job_id]["status"] = "evaluating"
        evaluator = DrugTargetEvaluator()
        
        if hasattr(germinal_center, "population") and germinal_center.population:
            evaluation_results = evaluator.evaluate_population(
                germinal_center.population,
                drug_targets
            )
        else:
            evaluation_results = {"error": "No population found"}
        
        # Store results
        job_results[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "request": request.dict(),
            "evolution_history": evolution_history,
            "evaluation_results": evaluation_results,
            "final_generation": running_jobs[job_id]["current_generation"],
            "elapsed_time": (datetime.now() - running_jobs[job_id]["start_time"]).total_seconds(),
            "completed_at": datetime.now().isoformat()
        }
        
        running_jobs[job_id]["status"] = "completed"
        logger.info(f"Drug discovery job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in drug discovery job {job_id}: {str(e)}")
        running_jobs[job_id]["status"] = "failed"
        running_jobs[job_id]["error"] = str(e)
        job_results[job_id] = {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "TE-AI Evolutionary Algorithm API",
        "version": "1.0.0",
        "endpoints": {
            "POST /evolution/start": "Start a new evolution job",
            "GET /evolution/status/{job_id}": "Get job status",
            "GET /evolution/results/{job_id}": "Get job results",
            "DELETE /evolution/cancel/{job_id}": "Cancel a running job",
            "GET /evolution/jobs": "List all jobs",
            "WS /evolution/stream/{job_id}": "Stream real-time updates"
        }
    }


@app.post("/evolution/start", response_model=EvolutionResponse)
async def start_evolution(
    request: EvolutionRequest,
    background_tasks: BackgroundTasks
):
    """Start a new evolution job"""
    job_id = str(uuid.uuid4())
    
    # Initialize job tracking
    running_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "current_generation": 0,
        "total_generations": request.generations,
        "population_size": 0,
        "best_fitness": 0.0,
        "diversity_score": 0.0,
        "created_at": datetime.now()
    }
    
    # Start background task
    if request.task_type == "drug_discovery":
        background_tasks.add_task(
            run_drug_discovery_evolution,
            job_id,
            request
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown task type: {request.task_type}")
    
    return EvolutionResponse(
        job_id=job_id,
        status="started",
        message=f"Evolution job started with {request.generations} generations",
        created_at=datetime.now()
    )


@app.get("/evolution/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a running or completed job"""
    if job_id in running_jobs:
        job = running_jobs[job_id]
        return JobStatus(
            job_id=job_id,
            status=job["status"],
            current_generation=job.get("current_generation", 0),
            total_generations=job.get("total_generations", 0),
            population_size=job.get("population_size", 0),
            best_fitness=job.get("best_fitness", 0.0),
            diversity_score=job.get("diversity_score", 0.0),
            elapsed_time=(datetime.now() - job["created_at"]).total_seconds(),
            metrics=job.get("metrics"),
            error=job.get("error")
        )
    elif job_id in job_results:
        result = job_results[job_id]
        return JobStatus(
            job_id=job_id,
            status=result["status"],
            current_generation=result.get("final_generation", 0),
            total_generations=result["request"]["generations"],
            population_size=0,
            best_fitness=0.0,
            diversity_score=0.0,
            elapsed_time=result.get("elapsed_time", 0.0),
            error=result.get("error")
        )
    else:
        raise HTTPException(status_code=404, detail="Job not found")


@app.get("/evolution/results/{job_id}")
async def get_job_results(job_id: str):
    """Get the results of a completed job"""
    if job_id not in job_results:
        if job_id in running_jobs:
            raise HTTPException(status_code=202, detail="Job still running")
        else:
            raise HTTPException(status_code=404, detail="Job not found")
    
    return job_results[job_id]


@app.delete("/evolution/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    if job_id not in running_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    running_jobs[job_id]["cancelled"] = True
    return {"message": f"Job {job_id} cancelled"}


@app.get("/evolution/jobs")
async def list_jobs():
    """List all jobs with their current status"""
    all_jobs = []
    
    # Add running jobs
    for job_id, job in running_jobs.items():
        all_jobs.append({
            "job_id": job_id,
            "status": job["status"],
            "created_at": job["created_at"].isoformat(),
            "current_generation": job.get("current_generation", 0),
            "total_generations": job.get("total_generations", 0)
        })
    
    # Add completed jobs
    for job_id, result in job_results.items():
        if job_id not in running_jobs:
            all_jobs.append({
                "job_id": job_id,
                "status": result["status"],
                "created_at": result.get("created_at", "unknown"),
                "completed_at": result.get("completed_at", "unknown"),
                "final_generation": result.get("final_generation", 0)
            })
    
    return {"jobs": all_jobs, "total": len(all_jobs)}


@app.websocket("/evolution/stream/{job_id}")
async def websocket_stream(websocket: WebSocket, job_id: str):
    """Stream real-time updates for a running job"""
    await websocket.accept()
    
    if job_id not in running_jobs:
        await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return
    
    try:
        while job_id in running_jobs and running_jobs[job_id]["status"] not in ["completed", "failed"]:
            # Send current status
            status_update = {
                "type": "status",
                "data": {
                    "job_id": job_id,
                    "status": running_jobs[job_id]["status"],
                    "current_generation": running_jobs[job_id].get("current_generation", 0),
                    "population_size": running_jobs[job_id].get("population_size", 0),
                    "best_fitness": running_jobs[job_id].get("best_fitness", 0.0),
                    "diversity_score": running_jobs[job_id].get("diversity_score", 0.0),
                    "metrics": running_jobs[job_id].get("metrics", {})
                }
            }
            await websocket.send_json(status_update)
            
            # Wait before next update
            await asyncio.sleep(1.0)
        
        # Send final status
        if job_id in job_results:
            await websocket.send_json({
                "type": "completed",
                "data": job_results[job_id]
            })
        
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
        "active_jobs": len([j for j in running_jobs.values() if j["status"] not in ["completed", "failed"]])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)