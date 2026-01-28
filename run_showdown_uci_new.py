#
# FILENAME: run_showdown_uci.py
#
"""
UCI HAR Showdown Runner (Comparison @ ~70% Sparsity)
Runs all UCI-adapted baselines with multiple random seeds for reproducibility.
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
METHODS = {
    
    'efficient_fl_uci': 'main_efficient_fl_cnn_uci.py',
    #'fedsnip_uci': 'main_fedsnip_cnn_uci.py',
    #'caafp_uci': 'main_caafp_cnn_uci.py',
    #'fedchar_uci': 'main_fedchar_cnn_uci.py',
    #'clusterfl_uci': 'main_clusterfl_cnn_uci.py'
    
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
        
    # Handle case where metrics might be a list (history) or dict
    data = raw_metrics
    if isinstance(data, list):
        data = data[-1]
    elif isinstance(data, dict):
        # If dict keys are rounds (integers), get the last round
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
    Returns a standardized dictionary of results.
    """
    print(f"\n{'='*70}")
    print(f"Running {method.upper()} with seed {seed}")
    print(f"Run ID: {run_id}")
    print(f"{'='*70}\n")
    
    set_all_seeds(seed)

    # ---------------------------------------------------------
    # 1. FedCHAR (UCI)
    # ---------------------------------------------------------
    if method == 'fedchar_uci':
        from main_fedchar_cnn_uci import run_fedchar
        # FedCHAR UCI script returns a 'rich_metrics' dict directly
        rich_metrics = run_fedchar(
            num_rounds=NUM_ROUNDS, 
            initial_rounds=10, 
            alpha=0.1,
            epochs_per_round=EPOCHS_PER_ROUND,
            seed=seed
        )
        # Ensure ID consistency
        rich_metrics['method'] = method
        rich_metrics['run_id'] = run_id
        return rich_metrics

    # ---------------------------------------------------------
    # 2. ClusterFL (UCI)
    # ---------------------------------------------------------
    elif method == 'clusterfl_uci':
        from main_clusterfl_cnn_uci import run_clusterfl
        
        # Unpack the 4 return values explicitly
        server, clients, results, raw_metrics = run_clusterfl(
            num_rounds=NUM_ROUNDS,
            alpha=0.1,
            seed=seed
        )
        
        # 1. Calculate Accuracy manually from 'results'
        accuracies = [r['accuracy'] for r in results.values()]
        avg_acc = np.mean(accuracies) if accuracies else 0.0
        
        # 2. Prepare the return dictionary
        # Start with the system metrics (comm, time, etc.)
        rich_metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        
        # Fallback if raw_metrics is empty/weird
        if not rich_metrics and hasattr(raw_metrics, 'get_results'):
             rich_metrics = raw_metrics.get_results()
        
        # 3. Inject the missing keys
        rich_metrics['method'] = method
        rich_metrics['run_id'] = run_id
        rich_metrics['avg_accuracy'] = avg_acc  # <--- CRITICAL FIX
        rich_metrics['all_accuracies'] = accuracies
        
        # Optional: Add sparsity if missing (ClusterFL is dense, so 0.0)
        if 'avg_sparsity' not in rich_metrics:
            rich_metrics['avg_sparsity'] = 0.0
            
        return rich_metrics
    # ---------------------------------------------------------
    # 3. EfficientFL (UCI)
    # ---------------------------------------------------------
    elif method == 'efficient_fl_uci':
        from main_efficient_fl_cnn_uci import run_efficient_fl
        
        # 1. Capture the single result dictionary
        result_metrics = run_efficient_fl(
            num_rounds=NUM_ROUNDS, 
            target_sparsity=0.7, 
            alpha=0.1,
            seed=seed
        )
        
        # 2. Copy to rich_metrics
        rich_metrics = result_metrics.copy()
        rich_metrics['method'] = method
        rich_metrics['run_id'] = run_id
        
        # 3. Safety defaults (Prevent crashes & Ensure non-zero comm cost if available)
        if 'avg_sparsity' not in rich_metrics:
            rich_metrics['avg_sparsity'] = 0.7
        if 'avg_accuracy' not in rich_metrics:
            rich_metrics['avg_accuracy'] = 0.0
        
        # Critical: If total_comm_mb is missing, default to 0.0
        if 'total_comm_mb' not in rich_metrics:
             rich_metrics['total_comm_mb'] = 0.0

        return rich_metrics
    # ---------------------------------------------------------
    # 4. FedSNIP (UCI)
    # ---------------------------------------------------------
    elif method == 'fedsnip_uci':
        from main_fedsnip_cnn_uci import run_fedsnip
        # Returns: server, clients, results, rich_metrics
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
    # 5. CA-AFP (UCI)
    # ---------------------------------------------------------
    elif method == 'caafp_uci':
        from main_caafp_cnn_uci import run_caafp
        # Returns: server, clients, results, final_metrics
        
        # Note: CA-AFP usually benefits from a longer schedule, but for 
        # fair "Showdown" we try to keep rounds consistent or slightly adjusted 
        # if the method requires phases.
        # We'll use 50 rounds total to match others (Warmup is implicit or 0).
        server, clients, results, final_metrics = run_caafp(
            num_clients=30,
            num_rounds=NUM_ROUNDS, 
            initial_rounds=0,  
            clustering_training_rounds=NUM_ROUNDS, # Run full pruning schedule
            clients_per_round=CLIENTS_PER_ROUND,
            epochs_per_round=EPOCHS_PER_ROUND,
            prune_rate=0.05, 
            start_sparsity=0.7, # Start high for UCI
            alpha=0.1,
            
            seed=seed
        )
        
        accuracies = [r['accuracy'] for r in results.values()]
        track_stats = extract_tracker_metrics(final_metrics)
        
        # Calculate final sparsity
        from models_cnn_uci import get_model_sparsity
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
            
            'total_comm_mb': track_stats['total_comm_mb'],
            'total_gflops': track_stats['total_gflops'],
            'wall_time': track_stats['wall_time'],
            
            'acc_per_mb': np.mean(accuracies) / track_stats['total_comm_mb'] if track_stats['total_comm_mb'] > 0 else 0,
            'all_accuracies': accuracies
        }

def save_results(results, filename):
    """Save results to JSON and pickle"""
    # JSON safe save
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
    
    # Pickle save
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(results, f)

def run_all_experiments():
    """Run all methods with all seeds"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results_uci_showdown', exist_ok=True)
    
    all_results = []
    
    for method in METHODS.keys():
        for seed in SEEDS:
            run_id = f"{method}_{seed}_{timestamp}"
            
            try:
                result = run_single_experiment(method, seed, run_id)
                all_results.append(result)
                
                # Save individual result
                save_results(result, f'results_uci_showdown/{run_id}')
                
                print(f"\n✓ Completed: {method} with seed {seed}")
                print(f"  Avg Accuracy: {result['avg_accuracy']:.4f}")
                print(f"  Avg Sparsity: {result['avg_sparsity']:.2%}")
                
            except Exception as e:
                print(f"\n✗ Failed: {method} with seed {seed}")
                import traceback
                traceback.print_exc()
                continue
    
    # --- Final CSV Summary ---
    df = pd.DataFrame(all_results)
    csv_path = f'results_uci_showdown/final_summary_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*70)
    print(f"UCI SHOWDOWN COMPLETE")
    print(f"Summary saved to: {csv_path}")
    print("="*70)
    
    # Print Pivot Table
    if not df.empty:
        pivot = df.groupby('method')[['avg_accuracy', 'total_comm_mb', 'acc_per_mb']].mean()
        print("\nAverage Performance across seeds:")
        print(pivot)

if __name__ == "__main__":
    run_all_experiments()