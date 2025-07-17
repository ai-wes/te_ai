"""
Example client for TE-AI Evolutionary API
=========================================

Demonstrates how to interact with the API for drug discovery tasks.
"""

import requests
import json
import time
import asyncio
import websockets
from typing import Dict, Any


class TEAIClient:
    """Client for interacting with TE-AI Evolutionary API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def start_drug_discovery(
        self, 
        generations: int = 50,
        target_proteins: list = None,
        disease_focus: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Start a drug discovery evolution job"""
        
        payload = {
            "task_type": "drug_discovery",
            "generations": generations,
            "target_proteins": target_proteins,
            "disease_focus": disease_focus,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/evolution/start",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a job"""
        response = requests.get(
            f"{self.base_url}/evolution/status/{job_id}"
        )
        response.raise_for_status()
        return response.json()
    
    def get_results(self, job_id: str) -> Dict[str, Any]:
        """Get the results of a completed job"""
        response = requests.get(
            f"{self.base_url}/evolution/results/{job_id}"
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(
        self, 
        job_id: str, 
        poll_interval: float = 5.0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Wait for a job to complete and return results"""
        
        while True:
            status = self.get_status(job_id)
            
            if verbose:
                print(f"Generation {status['current_generation']}/{status['total_generations']} - "
                      f"Fitness: {status['best_fitness']:.4f}, "
                      f"Diversity: {status['diversity_score']:.4f}")
            
            if status['status'] in ['completed', 'failed']:
                break
                
            time.sleep(poll_interval)
        
        return self.get_results(job_id)
    
    async def stream_updates(self, job_id: str):
        """Stream real-time updates via WebSocket"""
        uri = f"ws://localhost:8000/evolution/stream/{job_id}"
        
        async with websockets.connect(uri) as websocket:
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if data.get("type") == "status":
                        status = data["data"]
                        print(f"Gen {status['current_generation']}: "
                              f"Fitness={status['best_fitness']:.4f}, "
                              f"Diversity={status['diversity_score']:.4f}")
                    elif data.get("type") == "completed":
                        print("Evolution completed!")
                        return data["data"]
                    elif "error" in data:
                        print(f"Error: {data['error']}")
                        break
                        
                except websockets.exceptions.ConnectionClosed:
                    break


def example_drug_discovery():
    """Example: Run drug discovery for cancer targets"""
    
    client = TEAIClient()
    
    print("Starting drug discovery evolution...")
    
    # Start a job targeting specific cancer-related proteins
    response = client.start_drug_discovery(
        generations=20,
        target_proteins=["EGFR", "BRAF", "KRAS", "PIK3CA"],
        disease_focus="cancer",
        population_size=64,
        enable_quantum_dreams=True,
        enable_drug_genes=True
    )
    
    job_id = response["job_id"]
    print(f"Job started: {job_id}")
    
    # Wait for completion with status updates
    print("\nEvolution progress:")
    results = client.wait_for_completion(job_id, poll_interval=2.0)
    
    # Analyze results
    if results["status"] == "completed":
        print("\n✅ Evolution completed successfully!")
        
        eval_results = results.get("evaluation_results", {})
        if "top_drug_candidates" in eval_results:
            print("\nTop drug candidates found:")
            for i, candidate in enumerate(eval_results["top_drug_candidates"][:5]):
                print(f"{i+1}. Cell {candidate['cell_id']}: "
                      f"Druggability={candidate['druggability_score']:.3f}")
        
        # Show evolution statistics
        history = results.get("evolution_history", [])
        if history:
            final_gen = history[-1]
            print(f"\nFinal generation statistics:")
            print(f"  Population size: {final_gen.get('population_size', 'N/A')}")
            print(f"  Best fitness: {final_gen.get('best_fitness', 'N/A')}")
            print(f"  Diversity: {final_gen.get('diversity', 'N/A')}")
            print(f"  Active genes: {final_gen.get('active_genes', 'N/A')}")
    else:
        print(f"\n❌ Job failed: {results.get('error', 'Unknown error')}")


def example_with_omics_data():
    """Example: Use custom omics data for drug discovery"""
    
    client = TEAIClient()
    
    # Prepare omics data
    omics_data = {
        "gene_expression": {
            "EGFR": [5.2, 6.1, 4.8, 7.2],  # Expression across samples
            "BRAF": [3.1, 3.5, 3.2, 3.0],
            "KRAS": [4.5, 4.8, 5.1, 4.3]
        },
        "mutations": [
            {"Gene": "EGFR", "Position": 858, "WT": "L", "Mutant": "R"},
            {"Gene": "BRAF", "Position": 600, "WT": "V", "Mutant": "E"}
        ],
        "disease_associations": {
            "EGFR": 0.85,
            "BRAF": 0.92,
            "KRAS": 0.78
        }
    }
    
    print("Starting evolution with custom omics data...")
    
    response = client.start_drug_discovery(
        generations=30,
        omics_data=omics_data,
        disease_focus="melanoma",
        population_size=100
    )
    
    job_id = response["job_id"]
    print(f"Job started: {job_id}")
    
    # Stream real-time updates
    print("\nStreaming real-time updates:")
    asyncio.run(client.stream_updates(job_id))


async def example_async_monitoring():
    """Example: Monitor multiple jobs asynchronously"""
    
    client = TEAIClient()
    
    # Start multiple jobs
    jobs = []
    diseases = ["cancer", "alzheimer", "diabetes"]
    
    for disease in diseases:
        response = client.start_drug_discovery(
            generations=15,
            disease_focus=disease,
            population_size=50
        )
        jobs.append({
            "job_id": response["job_id"],
            "disease": disease
        })
        print(f"Started job for {disease}: {response['job_id']}")
    
    # Monitor all jobs concurrently
    async def monitor_job(job_info):
        job_id = job_info["job_id"]
        disease = job_info["disease"]
        
        while True:
            status = client.get_status(job_id)
            print(f"[{disease}] Gen {status['current_generation']}/{status['total_generations']}")
            
            if status['status'] in ['completed', 'failed']:
                break
                
            await asyncio.sleep(3.0)
        
        return disease, client.get_results(job_id)
    
    # Run monitoring tasks concurrently
    results = await asyncio.gather(*[monitor_job(job) for job in jobs])
    
    # Summarize results
    print("\n=== Summary ===")
    for disease, result in results:
        if result["status"] == "completed":
            eval_results = result.get("evaluation_results", {})
            n_candidates = len(eval_results.get("top_drug_candidates", []))
            print(f"{disease}: Found {n_candidates} drug candidates")
        else:
            print(f"{disease}: Failed - {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "omics":
        example_with_omics_data()
    elif len(sys.argv) > 1 and sys.argv[1] == "async":
        asyncio.run(example_async_monitoring())
    else:
        example_drug_discovery()