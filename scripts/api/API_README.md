# TE-AI Evolutionary Algorithm API

## Overview

The TE-AI API provides a RESTful interface for running evolutionary drug discovery and optimization tasks using the Transposable Element AI system. It supports asynchronous job execution, real-time monitoring via WebSockets, and flexible configuration options.

## Features

- **Asynchronous Job Processing**: Submit evolution jobs that run in the background
- **Real-time Updates**: Stream evolution progress via WebSocket connections
- **Multiple Task Types**: Support for drug discovery, protein optimization, and more
- **Flexible Input**: Use TCGA data, custom omics data, or auto-generated targets
- **Comprehensive Results**: Get detailed evolution history and evaluation metrics

## Installation

```bash
# Install required dependencies
pip install fastapi uvicorn websockets pandas

# Navigate to the API directory
cd scripts/api

# Start the API server
python evolutionary_api.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Core Endpoints

1. **Start Evolution Job**
   ```
   POST /evolution/start
   ```
   Start a new evolution job with specified parameters.

2. **Get Job Status**
   ```
   GET /evolution/status/{job_id}
   ```
   Check the current status and progress of a job.

3. **Get Job Results**
   ```
   GET /evolution/results/{job_id}
   ```
   Retrieve the complete results of a finished job.

4. **Cancel Job**
   ```
   DELETE /evolution/cancel/{job_id}
   ```
   Cancel a running job.

5. **List All Jobs**
   ```
   GET /evolution/jobs
   ```
   Get a list of all jobs and their statuses.

6. **Stream Updates**
   ```
   WS /evolution/stream/{job_id}
   ```
   WebSocket endpoint for real-time job updates.

### Utility Endpoints

- `GET /` - API information and available endpoints
- `GET /health` - Health check and system status

## Usage Examples

### Basic Drug Discovery

```python
import requests

# Start a drug discovery job
response = requests.post("http://localhost:8000/evolution/start", json={
    "task_type": "drug_discovery",
    "generations": 50,
    "target_proteins": ["EGFR", "BRAF", "KRAS"],
    "disease_focus": "cancer",
    "population_size": 64
})

job_id = response.json()["job_id"]

# Check status
status = requests.get(f"http://localhost:8000/evolution/status/{job_id}").json()
print(f"Generation: {status['current_generation']}/{status['total_generations']}")
print(f"Best fitness: {status['best_fitness']}")

# Get results when complete
results = requests.get(f"http://localhost:8000/evolution/results/{job_id}").json()
```

### Using Custom Omics Data

```python
# Prepare omics data
omics_data = {
    "gene_expression": {
        "GENE1": [5.2, 6.1, 4.8],
        "GENE2": [3.1, 3.5, 3.2]
    },
    "mutations": [
        {"Gene": "GENE1", "Position": 100, "WT": "A", "Mutant": "T"}
    ],
    "disease_associations": {
        "GENE1": 0.85,
        "GENE2": 0.65
    }
}

# Start job with omics data
response = requests.post("http://localhost:8000/evolution/start", json={
    "task_type": "drug_discovery",
    "generations": 30,
    "omics_data": omics_data,
    "disease_focus": "custom_disease"
})
```

### Real-time Monitoring with WebSocket

```python
import asyncio
import websockets
import json

async def monitor_evolution(job_id):
    uri = f"ws://localhost:8000/evolution/stream/{job_id}"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data["type"] == "status":
                status = data["data"]
                print(f"Gen {status['current_generation']}: "
                      f"Fitness={status['best_fitness']:.4f}")
            elif data["type"] == "completed":
                print("Evolution completed!")
                break

# Run monitoring
asyncio.run(monitor_evolution("your-job-id"))
```

## Request/Response Models

### EvolutionRequest
```json
{
    "task_type": "drug_discovery",
    "generations": 50,
    "population_size": 64,
    "target_proteins": ["EGFR", "BRAF"],
    "disease_focus": "cancer",
    "tcga_samples": ["sample1", "sample2"],
    "omics_data": {...},
    "enable_quantum_dreams": true,
    "enable_drug_genes": true
}
```

### JobStatus Response
```json
{
    "job_id": "uuid",
    "status": "evolving",
    "current_generation": 25,
    "total_generations": 50,
    "population_size": 64,
    "best_fitness": 0.875,
    "diversity_score": 4.2,
    "elapsed_time": 120.5,
    "metrics": {...}
}
```

### Evolution Results
```json
{
    "job_id": "uuid",
    "status": "completed",
    "request": {...},
    "evolution_history": [...],
    "evaluation_results": {
        "top_drug_candidates": [...],
        "druggability_scores": {...},
        "binding_profiles": {...}
    },
    "final_generation": 50,
    "elapsed_time": 300.2,
    "completed_at": "2024-01-15T10:30:00"
}
```

## Running with Docker

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run API
CMD ["python", "-m", "uvicorn", "evolutionary_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t te-ai-api .
docker run -p 8000:8000 -v /data:/data te-ai-api
```

## Performance Considerations

1. **GPU Acceleration**: The API automatically uses GPU if available for faster evolution
2. **Batch Processing**: Multiple antigens are processed in parallel for efficiency
3. **Memory Management**: Large populations may require significant GPU memory
4. **Concurrent Jobs**: Multiple jobs can run simultaneously (limited by system resources)

## API Authentication (Future Enhancement)

Currently, the API does not require authentication. For production use, consider adding:
- API key authentication
- JWT tokens for session management
- Rate limiting
- User quotas

## Error Handling

The API returns standard HTTP status codes:
- `200` - Success
- `202` - Job accepted/still running
- `400` - Bad request (invalid parameters)
- `404` - Job or resource not found
- `500` - Internal server error

Error responses include detailed messages:
```json
{
    "detail": "Error description"
}
```

## Monitoring and Logging

- Logs are written to the console and can be redirected to files
- Use the `/health` endpoint for monitoring system status
- Job history is maintained in memory (consider persistent storage for production)

## Advanced Features

### Custom Evolution Strategies
Extend the API to support custom evolution strategies by modifying the germinal center configuration.

### Batch Job Submission
Submit multiple jobs with different parameters for parameter sweeping:
```python
for pop_size in [50, 100, 200]:
    for generations in [20, 50, 100]:
        requests.post("/evolution/start", json={
            "population_size": pop_size,
            "generations": generations,
            ...
        })
```

### Result Visualization
Results can be visualized using the TE-AI visualization dashboard or exported for analysis in external tools.

## Support

For issues or questions:
1. Check the API logs for detailed error messages
2. Verify CUDA/GPU availability for performance issues
3. Ensure sufficient memory for large population sizes
4. Review the main TE-AI documentation for algorithm details