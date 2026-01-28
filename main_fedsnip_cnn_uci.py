#
# FILENAME: main_fedsnip_cnn_uci.py
#
import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime

# --- CHANGED: Use UCI Data Loader ---
from data_loader_uci import load_uci_har_dataset, create_natural_user_split, create_tf_datasets
# --- CHANGED: Use UCI Model ---
from models_cnn_uci import create_cnn_model
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

# --- FedSNIP Server ---
class FedSnipServer:
    def __init__(self, target_sparsity=0.7):
        # --- CHANGED: UCI Input Shape ---
        self.global_model = create_cnn_model(input_shape=(128, 9))
        self.target_sparsity = target_sparsity
        self.global_masks = []
        
    def aggregate_sensitivities_and_prune(self, client_sensitivity_lists):
        """
        [Round 0] Aggregates sensitivity scores by INDEX.
        Computes global threshold and creates binary masks.
        """
        print(f"  Aggregating sensitivities from {len(client_sensitivity_lists)} clients...")
        
        # 1. Sum sensitivities element-wise across clients
        avg_sensitivities = []
        for layer_scores in zip(*client_sensitivity_lists):
            summed = np.sum(np.stack(layer_scores), axis=0)
            avg_sensitivities.append(summed)
        
        # 2. Flatten ONLY PRUNABLE scores (Kernels) to find threshold
        all_scores = []
        trainable_vars = self.global_model.trainable_variables
        
        for i, score in enumerate(avg_sensitivities):
            var_name = trainable_vars[i].name
            # Protect biases
            if 'bias' not in var_name:
                all_scores.append(score.flatten())
        
        all_scores = np.concatenate(all_scores)
        
        # 3. Determine Threshold
        threshold = np.percentile(all_scores, self.target_sparsity * 100)
        print(f"  SNIP Threshold (Top {100-self.target_sparsity*100:.1f}%): {threshold:.6f}")
        
        # 4. Create Binary Masks
        self.global_masks = []
        for i, score in enumerate(avg_sensitivities):
            var_name = trainable_vars[i].name
            if 'bias' in var_name:
                # Keep biases 100%
                self.global_masks.append(np.ones_like(score))
            else:
                # Prune weights
                mask = (score > threshold).astype(np.float32)
                self.global_masks.append(mask)
            
        # 5. Apply Mask to Initial Global Weights
        for i, var in enumerate(trainable_vars):
            m = self.global_masks[i]
            if var.shape != m.shape:
                raise ValueError(f"Shape mismatch at index {i} ({var.name})")
            var.assign(var * m)
        
        print("  Global Pruning Mask Created (Index-Based, Biases protected).")
        return self.global_masks

    def aggregate_weights(self, client_weights_list):
        """FedAvg + Mask Re-application"""
        avg_weights = [np.mean(np.stack(w), axis=0) for w in zip(*client_weights_list)]
        
        masked_avg_weights = []
        for i, w in enumerate(avg_weights):
            if i < len(self.global_masks):
                m = self.global_masks[i]
                w = w * m
            masked_avg_weights.append(w)
            
        self.global_model.set_weights(masked_avg_weights)

# --- FedSNIP Client ---
class FedSnipClient:
    def __init__(self, client_id, train_dataset, test_dataset):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        # --- CHANGED: UCI Input Shape ---
        self.model = create_cnn_model(input_shape=(128, 9)) 
        
    @tf.function
    def _compute_grads(self, x, y):
        with tf.GradientTape() as tape:
            preds = self.model(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, preds))
        return tape.gradient(loss, self.model.trainable_variables)

    def compute_snip_sensitivity(self, global_weights):
        """Calculates sensitivity = |grad * weight|"""
        self.model.set_weights(global_weights)
        
        # Get single batch
        x_batch, y_batch = next(iter(self.train_dataset.take(1)))
        
        # Compute Gradients
        grads = self._compute_grads(x_batch, y_batch)
        
        # Calculate scores
        sensitivities = []
        current_weights = self.model.get_weights() 
        
        for g, w in zip(grads, current_weights):
            if g is not None:
                sensitivities.append(np.abs(g.numpy() * w))
            else:
                sensitivities.append(np.zeros_like(w))
        
        return sensitivities

    def train(self, global_weights, masks, epochs=1):
        self.model.set_weights(global_weights)
        self.model.fit(self.train_dataset, epochs=epochs, verbose=0)
        
        # Force Mask
        final_weights = []
        current_weights = self.model.get_weights()
        for i, w in enumerate(current_weights):
            if i < len(masks):
                m = masks[i]
                w = w * m
            final_weights.append(w)
        
        return final_weights

    def fine_tune(self, global_weights, masks, epochs=3):
        self.model.set_weights(global_weights)
        self.model.fit(self.train_dataset, epochs=epochs, verbose=0)
        
        # Re-apply mask after fine-tuning
        final_weights = []
        current_weights = self.model.get_weights()
        for i, w in enumerate(current_weights):
            if i < len(masks):
                m = masks[i]
                w = w * m
            final_weights.append(w)
        self.model.set_weights(final_weights)

    def evaluate(self):
        loss, acc = self.model.evaluate(self.test_dataset, verbose=0)
        return {'accuracy': acc}

# --- Main Run ---
def run_fedsnip(num_clients=30, num_rounds=50, clients_per_round=10,alpha=0.1, 
                target_sparsity=0.7, seed=42):
    
    print("="*60)
    print(f"FedSNIP (UCI HAR) | Seed: {seed}")
    print(f"Target Sparsity: {target_sparsity:.0%}")
    print("="*60)
    
    # --- CHANGED: Robust Seeding ---
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()

    # --- CHANGED: Data Loading ---
    print("Loading UCI HAR dataset...")
    X_data, y_data, user_ids = load_uci_har_dataset()
    client_data = create_natural_user_split(X_data, y_data, user_ids)
    #from data_loader_uci import create_non_iid_data_split
    #client_data = create_non_iid_data_split(
     #   X_data, y_data, user_ids, 
      #  num_clients=30, # Match CA-AFP client count
       # alpha=0.1,      # <--- SET ALPHA TO 0.1
        #seed=seed
    #)
    num_clients = len(client_data)
    train_datasets, test_datasets = create_tf_datasets(client_data, batch_size=32)
    
    server = FedSnipServer(target_sparsity=target_sparsity)
    clients = {i: FedSnipClient(i, train_datasets[i], test_datasets[i]) for i in range(num_clients)}
    metrics = FLMetricsTracker(server.global_model, train_datasets, epochs_per_round=3)
    
    # --- Round 0: One-Shot Pruning ---
    print("\n--- Round 0: Generating SNIP Masks ---")
    
    all_clients = list(clients.keys())
    global_w = server.global_model.get_weights()
    
    sensitivities = []
    print(f"  Computing sensitivities on {len(all_clients)} clients...")
    for cid in all_clients:
        s = clients[cid].compute_snip_sensitivity(global_w)
        sensitivities.append(s)
    
    global_masks = server.aggregate_sensitivities_and_prune(sensitivities)
    
    # --- FL Loops ---
    print(f"\n--- Starting Sparse FL Training ({num_rounds} rounds) ---")
    
    for round_num in range(num_rounds):
        metrics.start_round()
        global_w = server.global_model.get_weights()
        selected = np.random.choice(list(clients.keys()), size=clients_per_round, replace=False)
        updates = []
        
        for cid in selected:
            w_new = clients[cid].train(global_w, global_masks, epochs=3)
            updates.append(w_new)
        
        server.aggregate_weights(updates)
        
        if (round_num + 1) % 10 == 0:
            print(f"Round {round_num + 1}/{num_rounds} complete.")
        metrics.end_round(clients_per_round, sparsity=target_sparsity)

    # --- Final Fine-Tuning & Eval ---
    print("\nFinal Fine-Tuning & Evaluation:")
    final_w = server.global_model.get_weights()
    results = {}
    
    for cid, client in clients.items():
        client.fine_tune(final_w, global_masks, epochs=3)
        res = client.evaluate()
        results[cid] = res
        
    accs = [r['accuracy'] for r in results.values()]
    final_metrics_fl = metrics.get_results()
    print(f"Avg Acc: {np.mean(accs):.4f}")
    
    # --- ADDED: Return Rich Metrics ---
    # Handle the metrics dict safely
    if isinstance(final_metrics_fl, list):
        final_metrics_fl = final_metrics_fl[-1]
    elif isinstance(final_metrics_fl, dict) and any(isinstance(k, int) for k in final_metrics_fl.keys()):
         last_round = max(final_metrics_fl.keys())
         final_metrics_fl = final_metrics_fl[last_round]

    rich_metrics = {
        'method': 'FedSNIP-UCI',
        'seed': seed,
        'run_id': f"fedsnip_uci_{seed}",
        'avg_accuracy': np.mean(accs),
        'std_dev': np.std(accs),
        'min_accuracy': np.min(accs),
        'max_accuracy': np.max(accs),
        'avg_sparsity': target_sparsity,
        
        'total_comm_mb': final_metrics_fl.get('total_comm_mb', 0), 
        'total_gflops': final_metrics_fl.get('total_gflops', 0),
        'wall_time': final_metrics_fl.get('total_training_time', 0),
        
        'acc_per_mb': np.mean(accs) / final_metrics_fl.get('total_comm_mb', 1) if final_metrics_fl.get('total_comm_mb', 0) > 0 else 0,
        'all_accuracies': accs
    }
    
    return server, clients, results, rich_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_sparsity', type=float, default=0.7)
    parser.add_argument('--exp_name', type=str, default="uci_fedsnip_baseline")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"log_{args.exp_name}_{timestamp}.txt"
    sys.stdout = Logger(log_filename)
    
    seeds = [42, 123, 456, 789, 1024]
    all_metrics_list = []
    
    for seed in seeds:
        print(f"\n\n{'#'*60}\nRUNNING SEED: {seed}\n{'#'*60}")
        
        _, _, _, result_dict = run_fedsnip(
            num_rounds=50, 
            target_sparsity=args.start_sparsity,
            seed=seed
        )
        all_metrics_list.append(result_dict)

    df_metrics = pd.DataFrame(all_metrics_list)
    metrics_csv = f"results_{args.exp_name}_{timestamp}.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    
    print("\n" + "="*60)
    print(f"SUMMARY (Avg across {len(seeds)} seeds)")
    print("="*60)
    print(f"Avg Accuracy: {df_metrics['avg_accuracy'].mean():.4f} +/- {df_metrics['avg_accuracy'].std():.4f}")
    print("="*60)