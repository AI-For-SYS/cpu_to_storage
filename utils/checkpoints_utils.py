import os
import json
from typing import Optional



# Helper functions for incremental result saving and resumption
def save_incremental_results(output_file: str, results: dict, append: bool = False):
    """Save results incrementally to a JSON file.
    
    Args:
        output_file: Path to the output JSON file
        results: Dictionary containing the results to save
        append: If True, merge with existing results (for resumption)
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if append and os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
            # Merge results - this will overwrite existing keys
            for key in results:
                if key in existing_results and isinstance(existing_results[key], dict) and isinstance(results[key], dict):
                    existing_results[key].update(results[key])
                else:
                    existing_results[key] = results[key]
            results = existing_results
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load existing results from {output_file}: {e}")
    
    # Write atomically using a temporary file
    temp_file = output_file + '.tmp'
    with open(temp_file, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(temp_file, output_file)

def load_existing_results(output_file: str) -> Optional[dict]:
    """Load existing results from a JSON file if it exists.
    
    Args:
        output_file: Path to the output JSON file
        
    Returns:
        Dictionary containing existing results, or None if file doesn't exist or is invalid
    """
    if not os.path.exists(output_file):
        return None
    
    try:
        with open(output_file, 'r') as f:
            results = json.load(f)
        return results
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load existing results from {output_file}: {e}")
        return None

def check_config_match(existing_config: dict, new_config: dict) -> bool:
    """Check if the configuration of existing results matches the new benchmark config.
    
    Args:
        existing_config: Configuration from existing results
        new_config: Configuration for the new benchmark
        
    Returns:
        True if configurations match, False otherwise
    """
    # Compare key configuration parameters
    key_params = ['buffer_size', 'num_iterations', 'block_sizes_mb', 'thread_counts', 
                  'num_blocks', 'total_data_size_gb', 'implementation']
    
    for param in key_params:
        if param in new_config:
            if param not in existing_config:
                return False
            if existing_config[param] != new_config[param]:
                return False
    
    return True

def get_completed_tests(results: dict, test_type: str) -> set:
    """Get the set of completed test combinations from existing results.
    
    Args:
        results: Dictionary containing existing results
        test_type: Either 'write' or 'read'
        
    Returns:
        Set of tuples representing completed (thread_count, block_size) combinations
    """
    completed = set()
    
    if test_type not in results:
        return completed
    
    test_results = results[test_type]
    
    # Handle different result structures
    for key1, value1 in test_results.items():
        if isinstance(value1, dict):
            for key2 in value1.keys():
                completed.add((key1, key2))
    
    return completed
