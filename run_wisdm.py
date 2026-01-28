#
# FILENAME: run_showdown_wisdm.py
#
"""
WISDM Showdown Runner (Comparison on Human Activity Recognition)
Runs all WISDM-adapted baselines with multiple random seeds.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from datetime import datetime
import pickle

# --- Configuration ---
SEEDS = [42, 123, 456, 789, 1024]
NUM_ROUNDS = 50 
CLIENTS_PER_ROUND = 10
EPOCHS_PER_ROUND = 3

# Map method names to their file (for reference/logging)
# CHANGED: Updated filenames to _wisdm versions
METHODS = {
    #'efficient_fl_wisdm': 'main_efficient_fl_cnn_wisdm.py',
    #'fedsnip_wisdm': 'main_fedsnip_cnn_wisdm.py',
    #'fedchar_wisdm': 'main_fedchar_cnn_wisdm.py',
    #'clusterfl_wisdm': 'main_clusterfl_cnn_wisdm.py',
    'caafp_wisdm': 'main_caafp_cnn_wisdm.py'
}

def set_all_seeds(seed):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Set all seeds to {seed}")

def extract_tracker_metrics(raw_metrics):
    """Safe extraction of metrics from FLMetricsTracker output"""
    if not raw_metrics:
        return {}
        
    data = raw_metrics
    if isinstance(data, list):
        data = data[-1]
    elif isinstance(data, dict):
        if any(isinstance(k, int) for k in data.keys()):
            last_round = max(data.keys())
            data = data[last_round]
            
    return {
        'total_comm_mb': data.get('total_comm_mb', 0),
        'total_gflops': data.get('total_gflops', 0),
        'wall_time': data.get('total_training_time', 0)
    }

def run_single_experiment(method, seed, run_id):
    """
    Run a single experiment with specified method and seed.
    """
    print(f"\n{'='*70}")
    print(f"Running {method.upper()} with seed {seed}")
    print(f"Run ID: {run_id}")
    print(f"{'='*70}\n")
    
    set_all_seeds(seed)

    # ---------------------------------------------------------
    # 1. FedCHAR (WISDM)
    # ---------------------------------------------------------
    if method == 'fedchar_wisdm':
        # CHANGED: Import from WISDM file
        from main_fedchar_cnn import run_fedchar
        
        rich_metrics = run_fedchar(
            num_rounds=NUM_ROUNDS, 
            initial_rounds=10, 
            alpha=0.1,
            epochs_per_round=EPOCHS_PER_ROUND,
            seed=seed
        )
        rich_metrics['method'] = method
        rich_metrics['run_id'] = run_id
        return rich_metrics

    # ---------------------------------------------------------
    # 2. ClusterFL (WISDM)
    # ---------------------------------------------------------
    elif method == 'clusterfl_wisdm':
        # CHANGED: Import from WISDM file
        from main_clusterfl_cnn import run_clusterfl
        
        # Note: We added 'seed' argument to run_clusterfl in the previous step
        server, clients, results, raw_metrics = run_clusterfl(
            num_rounds=NUM_ROUNDS,
            alpha=0.1,
            seed=seed 
        )
        
        accuracies = [r['accuracy'] for r in results.values()]
        avg_acc = np.mean(accuracies) if accuracies else 0.0
        
        rich_metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        if not rich_metrics and hasattr(raw_metrics, 'get_results'):
             rich_metrics = raw_metrics.get_results()
        
        rich_metrics['method'] = method
        rich_metrics['run_id'] = run_id
        rich_metrics['avg_accuracy'] = avg_acc
        rich_metrics['all_accuracies'] = accuracies
        
        if 'avg_sparsity' not in rich_metrics:
            rich_metrics['avg_sparsity'] = 0.0
            
        return rich_metrics

    # ---------------------------------------------------------
    # 3. EfficientFL (WISDM)
    # ---------------------------------------------------------
    elif method == 'efficient_fl_wisdm':
        # CHANGED: Import from WISDM file
        from main_efficient_fl_cnn import run_efficient_fl
        
        return_values = run_efficient_fl(
            num_rounds=NUM_ROUNDS, 
            target_sparsity=[0.7], 
            alpha=0.1,
            seed=seed
        )
        
        rich_metrics = {}
        found = False

        if isinstance(return_values, dict):
            if 'avg_accuracy' in return_values:
                rich_metrics = return_values
                found = True
            else:
                for key in ['metrics', 'rich_metrics', 'final_metrics', 'results']:
                    if key in return_values and isinstance(return_values[key], dict):
                        rich_metrics = return_values[key]
                        found = True
                        break
        
        elif isinstance(return_values, (list, tuple)):
            for item in reversed(return_values):
                if isinstance(item, dict) and 'avg_accuracy' in item:
                    rich_metrics = item
                    found = True
                    break
            
            if not found and len(return_values) >= 4:
                if isinstance(return_values[3], dict):
                    rich_metrics = return_values[3]
                    found = True

        if not found:
            print(f"[WARNING] Could not find metrics dict in EfficientFL return")
            rich_metrics = {'avg_accuracy': 0.0, 'total_comm_mb': 0.0, 'error': 'Metrics Not Found'}

        rich_metrics['method'] = method
        rich_metrics['run_id'] = run_id
        return rich_metrics

    # ---------------------------------------------------------
    # 4. FedSNIP (WISDM)
    # ---------------------------------------------------------
    elif method == 'fedsnip_wisdm':
        # CHANGED: Import from WISDM file
        from main_fedsnip_cnn import run_fedsnip
        
        _, _, _, rich_metrics = run_fedsnip(
            num_rounds=NUM_ROUNDS, 
            target_sparsity=0.7, 
            alpha=0.1,
            seed=seed
        )
        rich_metrics['method'] = method
        rich_metrics['run_id'] = run_id
        return rich_metrics

    # ---------------------------------------------------------
    # 5. CA-AFP (WISDM)
    # ---------------------------------------------------------
    elif method == 'caafp_wisdm':
        # CHANGED: Import from WISDM file
        from main_caafp_cnn import run_caafp
        
        # Num clients is handled dynamically inside main_caafp_cnn_wisdm
        server, clients, results, final_metrics = run_caafp(
            num_clients=30, # Argument kept for compatibility, but internal logic overwrites it
            num_rounds=NUM_ROUNDS, 
            initial_rounds=0,  
            clustering_training_rounds=NUM_ROUNDS, 
            clients_per_round=CLIENTS_PER_ROUND,
            epochs_per_round=EPOCHS_PER_ROUND,
            prune_rate=0.05, 
            start_sparsity=0.7, 
            alpha=0.1,
            seed=seed
        )
        
        accuracies = [r['accuracy'] for r in results.values()]
        track_stats = extract_tracker_metrics(final_metrics)
        
        # CHANGED: Import sparsity helper from WISDM models
        from models_cnn import get_model_sparsity
        
        sparsities = [get_model_sparsity(m) for m in server.final_pruned_models.values()]
        avg_sparsity = np.mean(sparsities) if sparsities else 0.0
        
        return {
            'method': method,
            'seed': seed,
            'run_id': run_id,
            'avg_accuracy': np.mean(accuracies),
            'std_dev': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'avg_sparsity': avg_sparsity,
            
            'total_comm_mb': track_stats.get('total_comm_mb', 0),
            'total_gflops': track_stats.get('total_gflops', 0),
            'wall_time': track_stats.get('wall_time', 0),
            
            'acc_per_mb': np.mean(accuracies) / track_stats.get('total_comm_mb', 1e-9),
            'all_accuracies': accuracies
        }

def save_results(results, filename):
    """Save results to JSON and pickle"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename + '.json', 'w') as f:
        results_json = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                results_json[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                results_json[k] = float(v)
            elif isinstance(v, list):
                results_json[k] = v
            else:
                results_json[k] = str(v)
        json.dump(results_json, f, indent=2)
    
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(results, f)

def run_all_experiments():
    """Run all methods with all seeds"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # CHANGED: New output directory
    output_dir = 'results_wisdm_showdown'
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for method in METHODS.keys():
        for seed in SEEDS:
            run_id = f"{method}_{seed}_{timestamp}"
            
            try:
                result = run_single_experiment(method, seed, run_id)
                all_results.append(result)
                
                # Save individual result
                save_results(result, f'{output_dir}/{run_id}')
                
                print(f"\n✓ Completed: {method} with seed {seed}")
                print(f"  Avg Accuracy: {result['avg_accuracy']:.4f}")
                print(f"  Avg Sparsity: {result.get('avg_sparsity', 0):.2%}")
                
            except Exception as e:
                print(f"\n✗ Failed: {method} with seed {seed}")
                import traceback
                traceback.print_exc()
                continue
    
    # --- Final CSV Summary ---
    df = pd.DataFrame(all_results)
    csv_path = f'{output_dir}/final_summary_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*70)
    print(f"WISDM SHOWDOWN COMPLETE")
    print(f"Summary saved to: {csv_path}")
    print("="*70)
    
    if not df.empty:
        pivot = df.groupby('method')[['avg_accuracy', 'total_comm_mb', 'acc_per_mb']].mean()
        print("\nAverage Performance across seeds:")
        print(pivot)

if __name__ == "__main__":
    run_all_experiments()
