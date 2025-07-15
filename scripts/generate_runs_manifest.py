#!/usr/bin/env python3
"""
Generate a manifest of all visualization runs for the dropdown menu
"""

import os
import json
import glob

def generate_manifest():
    """Scan visualization_data directory and create a manifest of all runs"""
    
    viz_dir = "visualization_data"
    if not os.path.exists(viz_dir):
        print(f"No {viz_dir} directory found")
        return
    
    runs = []
    
    # Find all run directories
    for run_dir in sorted(glob.glob(os.path.join(viz_dir, "*"))):
        if os.path.isdir(run_dir):
            run_id = os.path.basename(run_dir)
            
            # Check for metadata.json
            metadata_file = os.path.join(run_dir, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Count generation files
                gen_files = glob.glob(os.path.join(run_dir, "generation_*.json"))
                
                runs.append({
                    'id': run_id,
                    'generations': metadata.get('current_generation', len(gen_files)),
                    'timestamp': metadata.get('timestamp', 0),
                    'total_files': len(gen_files)
                })
    
    # Write manifest
    manifest = {
        'runs': runs,
        'generated_at': os.path.getmtime(__file__)
    }
    
    manifest_path = os.path.join(viz_dir, 'runs_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Generated manifest with {len(runs)} runs at {manifest_path}")

if __name__ == "__main__":
    generate_manifest()