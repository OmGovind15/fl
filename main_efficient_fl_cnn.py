#
# FILENAME: main_efficient_fl_cnn_uci.py
#
import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd  # Added for CSV saving
from datetime import datetime

# --- CHANGED: Load from UCI Loader ---
from data_loader import load_wisdm_dataset, create_natural_user_split, create_tf_datasets
#from models_cnn import create_cnn_model, get_model_sparsity # Added get_model_sparsity
from models_cnn import create_cnn_model, get_model_sparsity
from metrics import FLMetricsTracker

# --- Logger Class ---
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message); self.logfile.write(message); self.flush()
    def flush(self):
        self.terminal.flush(); self.logfile.flush()
    def __del__(self):
        sys.stdout = self.terminal
        if self.logfile: self.logfile.close()

# --- Helper: Structural Pruning Builder (CNN) ---
def build_structurally_pruned_model(original_model, filters_map):
    """
    Rebuilds CNN with reduced filters.
    filters_map: {layer_name: num_filters_to_keep}
    """
    input_shape = original_model.input_shape[1:] 
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    
    # Iterate original layers to preserve order/naming
    for layer in original_model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            new_filters = filters_map.get(layer.name, layer.filters)
            x = tf.keras.layers.Conv1D(
                filters=new_filters,
                kernel_size=layer.kernel_size,
                activation=layer.activation,
                padding=layer.padding,
                name=layer.name
            )(x)
        elif isinstance(layer, tf.keras.layers.MaxPooling1D):
            x = tf.keras.layers.MaxPooling1D(pool_size=layer.pool_size)(x)
        elif isinstance(layer, tf.keras.layers.Flatten):
            x = tf.keras.layers.Flatten()(x)
        elif isinstance(layer, tf.keras.layers.Dropout):
            x = tf.keras.layers.Dropout(layer.rate)(x)
        elif isinstance(layer, tf.keras.layers.Dense):
            x = tf.keras.layers.Dense(
                layer.units, 
                activation=layer.activation, 
                name=layer.name
            )(x)
            
    new_model = tf.keras.Model(inputs=inputs, outputs=x)
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return new_model

# --- Server: Efficient FL (Wu et al.) ---
class EfficientFLServer:
    def __init__(self, num_clients, target_sparsity=0.7):
        tf.keras.backend.clear_session()
        # --- CHANGED: Input shape for UCI HAR (128, 9) ---
        self.global_model = create_cnn_model()
        
        # --- FIX: Ensure sparsity is a float, not a list ---
        if isinstance(target_sparsity, (list, tuple, np.ndarray)):
             self.target_sparsity = float(target_sparsity[0])
        else:
             self.target_sparsity = float(target_sparsity)
        
        self.keep_ratio = self._calculate_keep_ratio()
        
        self.global_layer_weights = {}
        for layer in self.global_model.layers:
            if layer.trainable_weights:
                self.global_layer_weights[layer.name] = [w.numpy() for w in layer.trainable_variables]
        
        self.active_indices = {} 
        self.filters_map = {} 

    def _calculate_keep_ratio(self):
        return 1.0 - self.target_sparsity

    def _find_preceding_conv_layer(self, current_layer_index):
        """Backtracks to find the last Conv1D layer (to determine input shapes for Dense)."""
        for i in range(current_layer_index - 1, -1, -1):
            layer = self.global_model.layers[i]
            if isinstance(layer, tf.keras.layers.Conv1D):
                return layer
        return None

    def prune_and_distribute(self):
        """
        [Algorithm 1] Structural Pruning for CNN
        """
        self.active_indices = {} 
        self.filters_map = {}
        
        # --- 1. Select Active Filters (L1 Norm) ---
        for layer in self.global_model.layers:
            if not isinstance(layer, tf.keras.layers.Conv1D): continue
            
            # Conv1D Kernel: (KernelSize, InputChannels, Filters)
            weights = self.global_layer_weights[layer.name]
            kernel = weights[0] 
            n_filters = kernel.shape[-1]
            
            # L1 Norm
            importances = np.sum(np.abs(kernel), axis=(0, 1))
            
            # Select Top-K
            n_keep = max(1, int(n_filters * self.keep_ratio))
            top_indices = np.argsort(importances)[-n_keep:]
            top_indices = np.sort(top_indices)
            
            self.active_indices[layer.name] = top_indices
            self.filters_map[layer.name] = len(top_indices)

        # --- 2. Build Subnet and Slice Weights ---
        pruned_model = build_structurally_pruned_model(self.global_model, self.filters_map)
        pruned_weights_map = {}
        
        prev_active_indices = None 
        
        for i, layer in enumerate(self.global_model.layers):
            if layer.name not in self.global_layer_weights: continue
            weights = self.global_layer_weights[layer.name]
            kernel, bias = weights[0], weights[1]
            
            if isinstance(layer, tf.keras.layers.Conv1D):
                active = self.active_indices[layer.name]
                
                # Slice Output
                sub_kernel = kernel[:, :, active]
                sub_bias = bias[active]
                
                # Slice Input
                if prev_active_indices is not None:
                    sub_kernel = sub_kernel[:, prev_active_indices, :]
                
                pruned_weights_map[layer.name] = [sub_kernel, sub_bias]
                prev_active_indices = active

            elif isinstance(layer, tf.keras.layers.Dense):
                # Flatten Mapping Logic
                if prev_active_indices is not None:
                    flatten_dim = kernel.shape[0]
                    
                    prev_conv = self._find_preceding_conv_layer(i)
                    if prev_conv is None:
                        n_total_prev = 64 
                    else:
                        n_total_prev = prev_conv.filters
                        
                    spatial_steps = flatten_dim // n_total_prev
                    
                    keep_row_indices = []
                    for s in range(spatial_steps):
                        for idx in prev_active_indices:
                            keep_row_indices.append(s * n_total_prev + idx)
                    
                    sub_kernel = kernel[keep_row_indices, :]
                else:
                    sub_kernel = kernel
                
                sub_bias = bias
                pruned_weights_map[layer.name] = [sub_kernel, sub_bias]
                prev_active_indices = None
        
        # Assign weights
        for layer in pruned_model.layers:
            if layer.name in pruned_weights_map:
                layer.set_weights(pruned_weights_map[layer.name])
                
        return pruned_model

    def aggregate_and_update(self, client_weights_list):
        """
        Aggregate and map back to global
        """
        dummy_pruned = build_structurally_pruned_model(self.global_model, self.filters_map)
        pruned_layers = [l for l in dummy_pruned.layers if l.trainable_weights]
        
        avg_weights_by_layer = []
        flat_idx = 0
        for layer in pruned_layers:
            n_vars = len(layer.trainable_weights)
            layer_updates = [cw[flat_idx : flat_idx + n_vars] for cw in client_weights_list]
            avg_w = [np.mean(np.stack(col), axis=0) for col in zip(*layer_updates)]
            avg_weights_by_layer.append(avg_w)
            flat_idx += n_vars

        # Update Global
        global_trainable = [l for l in self.global_model.layers if l.trainable_weights]
        prev_active_indices = None
        
        for j, layer in enumerate(global_trainable):
            if j >= len(avg_weights_by_layer): break
            
            g_weights = self.global_layer_weights[layer.name]
            p_kernel, p_bias = avg_weights_by_layer[j]
            
            g_kernel, g_bias = g_weights
            
            if isinstance(layer, tf.keras.layers.Conv1D):
                active = self.active_indices[layer.name]
                g_bias[active] = p_bias
                
                if prev_active_indices is not None:
                    idx_in = prev_active_indices[:, np.newaxis]
                    idx_out = active
                    g_kernel[:, idx_in, idx_out] = p_kernel
                else:
                    g_kernel[:, :, active] = p_kernel
                
                self.global_layer_weights[layer.name] = [g_kernel, g_bias]
                prev_active_indices = active

            elif isinstance(layer, tf.keras.layers.Dense):
                g_bias[:] = p_bias
                
                if prev_active_indices is not None:
                    full_idx = self.global_model.layers.index(layer)
                    prev_conv = self._find_preceding_conv_layer(full_idx)
                    n_total_prev = prev_conv.filters if prev_conv else 64
                    
                    flatten_dim = g_kernel.shape[0]
                    spatial_steps = flatten_dim // n_total_prev
                    
                    keep_row_indices = []
                    for s in range(spatial_steps):
                        for idx in prev_active_indices:
                            keep_row_indices.append(s * n_total_prev + idx)
                    
                    g_kernel[keep_row_indices, :] = p_kernel
                else:
                    g_kernel[:] = p_kernel
                    
                self.global_layer_weights[layer.name] = [g_kernel, g_bias]
                prev_active_indices = None

        # Sync Global Object
        for layer in self.global_model.layers:
            if layer.name in self.global_layer_weights:
                layer.set_weights(self.global_layer_weights[layer.name])

# --- Client ---
class EfficientFLClient:
    def __init__(self, client_id, train_dataset, test_dataset):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = None 

    def train(self, pruned_model_template, epochs=1):
        if self.model is None:
             self.model = tf.keras.models.clone_model(pruned_model_template)
             self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
             try:
                 self.model.set_weights(pruned_model_template.get_weights())
             except ValueError:
                 self.model = tf.keras.models.clone_model(pruned_model_template)
                 self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                 self.model.set_weights(pruned_model_template.get_weights())
        
        self.model.fit(self.train_dataset, epochs=epochs, verbose=0)
        return self.model.get_weights()

    def fine_tune(self, pruned_model_template, epochs=3):
        self.train(pruned_model_template, epochs=epochs)

    def evaluate(self):
        if self.model:
            res = self.model.evaluate(self.test_dataset, verbose=0)
            return {'accuracy': res[1]}
        return {'accuracy': 0.0}

# --- Main Run ---
def run_efficient_fl(num_clients=30, num_rounds=50, clients_per_round=10, 
                     target_sparsity=0.7, alpha=0.1, seed=42):
    
    # --- FIX: Sanitize target_sparsity input (handle list vs float) ---
    if isinstance(target_sparsity, (list, tuple, np.ndarray)):
        target_sparsity = float(target_sparsity[0])
    else:
        target_sparsity = float(target_sparsity)
    # ------------------------------------------------------------------

    print("="*60)
    print(f"Efficient FL (Wu et al.) [CNN + Structural] | Seed: {seed}")
    print(f"Target Sparsity: {target_sparsity:.0%}")
    print("="*60)
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()
    
    print("\n[INFO] Loading WISDM Data...")
    X_data, y_data, user_ids = load_wisdm_dataset() # uses default path

    # Use Natural User Split (1 User = 1 Client)
    client_data = create_natural_user_split(X_data, y_data, user_ids)
    
    # Update num_clients dynamically based on actual users found
    num_clients = len(client_data)
    print(f"[INFO] Total Clients: {num_clients}")

    # Create TF Datasets (using safe per-client scaling)
    train_datasets, test_datasets = create_tf_datasets(client_data, batch_size=32)
    
    server = EfficientFLServer(num_clients, target_sparsity=target_sparsity)
    clients = {i: EfficientFLClient(i, train_datasets[i], test_datasets[i]) for i in range(num_clients)}
    metrics = FLMetricsTracker(server.global_model, train_datasets, epochs_per_round=3)
    
    for round_num in range(num_rounds):
        metrics.start_round()
        pruned_model = server.prune_and_distribute()
        
        current_sparsity = get_model_sparsity(pruned_model)
        
        selected = np.random.choice(list(clients.keys()), size=clients_per_round, replace=False)
        updates = []
        for cid in selected:
            w = clients[cid].train(pruned_model, epochs=3)
            updates.append(w)
            
        server.aggregate_and_update(updates)
        
        # --- NEW (CORRECT) ---
        # Pass the target sparsity (e.g. 0.7) directly so the tracker knows
        # to calculate communication as "Full Model Size * (1 - 0.7)"
        metrics.end_round(len(selected), sparsity=target_sparsity)

        if (round_num + 1) % 10 == 0:
            print(f"Round {round_num + 1}/{num_rounds} complete.")

    print("\nFinal Fine-Tuning & Evaluation:")
    final_pruned_model = server.prune_and_distribute() 
    results = {}
    
    for cid, client in clients.items():
        client.fine_tune(final_pruned_model, epochs=3)
        res = client.evaluate()
        results[cid] = res
        
    accs = [r['accuracy'] for r in results.values()]
    
    # --- ROBUST METRIC EXTRACTION ---
    raw_metrics = metrics.get_results()
    final_metrics_fl = {} # Default to empty dict
    
    if raw_metrics:
        # 1. Get the last item regardless of list vs dict
        last_item = None
        if isinstance(raw_metrics, list):
            last_item = raw_metrics[-1]
        elif isinstance(raw_metrics, dict):
            try:
                last_round = max(raw_metrics.keys())
                last_item = raw_metrics[last_round]
            except ValueError:
                pass
        
        # 2. Safety Check: Only use it if it is actually a dictionary
        if isinstance(last_item, dict):
            final_metrics_fl = last_item
        else:
            # If metrics.py returned a float/int (e.g. just accuracy), 
            # we ignore it here because we calculated 'accs' manually above.
            pass
            
    print(f"Avg Acc: {np.mean(accs):.4f}")
    
    rich_metrics = {
        'method': 'EfficientFL-UCI',
        'seed': seed,
        'run_id': f"eff_uci_{seed}",
        'avg_accuracy': np.mean(accs),
        'std_dev': np.std(accs),
        'min_accuracy': np.min(accs),
        'max_accuracy': np.max(accs),
        'avg_sparsity': get_model_sparsity(final_pruned_model),
        
        'total_comm_mb': final_metrics_fl.get('total_comm_mb', 0), 
        'total_gflops': final_metrics_fl.get('total_gflops', 0),
        'wall_time': final_metrics_fl.get('total_training_time', 0),
        
        'acc_per_mb': np.mean(accs) / final_metrics_fl.get('total_comm_mb', 1) if final_metrics_fl.get('total_comm_mb', 0) > 0 else 0,
        
        'cluster_0_acc': 0, 
        'cluster_1_acc': 0,
        'cluster_2_acc': 0,
    }

    return rich_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_sparsity', type=float, default=0.7) # Target sparsity
    parser.add_argument('--exp_name', type=str, default="uci_efficient_baseline")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"log_{args.exp_name}_{timestamp}.txt"
    sys.stdout = Logger(log_filename)
    
    # --- CHANGED: Multi-Seed Loop for Comparison ---
    seeds = [42, 123, 456, 789, 1024]
    all_metrics_list = []
    
    for seed in seeds:
        print(f"\n\n{'#'*60}\nRUNNING SEED: {seed}\n{'#'*60}")
        
        result_dict = run_efficient_fl(
            num_rounds=40, 
            target_sparsity=args.start_sparsity, # EfficientFL uses this as fixed target
            seed=seed
        )
        all_metrics_list.append(result_dict)

    # --- SAVE METRICS TO CSV ---
    df_metrics = pd.DataFrame(all_metrics_list)
    metrics_csv = f"results_{args.exp_name}_{timestamp}.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"\n[SAVED] Metrics saved to: {metrics_csv}")
    
    # --- PRINT SUMMARY ---
    print("\n" + "="*60)
    print(f"SUMMARY (Avg across {len(seeds)} seeds)")
    print("="*60)
    print(f"Avg Accuracy: {df_metrics['avg_accuracy'].mean():.4f} +/- {df_metrics['avg_accuracy'].std():.4f}")
    print("="*60)