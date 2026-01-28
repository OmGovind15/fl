#
# FILENAME: main_fedchar_cnn_uci.py
#
import os
import sys
import numpy as np
# These must be set before TensorFlow is loaded by ANY file
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
import tensorflow as tf
import pandas as pd
from datetime import datetime
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
from data_loader import load_wisdm_dataset, create_natural_user_split, create_tf_datasets
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

# --- FedCHAR Server ---
class FedCHARServer:
    def __init__(self, num_clients, num_clusters=3):
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.global_model = create_cnn_model()
        self.cluster_models = {}
        self.client_clusters = {}  
        self.clustered = False

    def aggregate_weights(self, client_weights_list):
        avg_weights = [np.mean(np.stack(w), axis=0) for w in zip(*client_weights_list)]
        return avg_weights

    def cluster_clients(self, client_updates):
        flattened_updates = []
        for update in client_updates:
            flattened = np.concatenate([u.flatten() for u in update])
            flattened_updates.append(flattened)
        flattened_updates = np.array(flattened_updates)

        n_clients = len(flattened_updates)
        distance_matrix = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(n_clients):
                if i != j:
                    distance_matrix[i, j] = cosine(flattened_updates[i], flattened_updates[j])
                else:
                    distance_matrix[i, j] = 0.0

        clustering = AgglomerativeClustering(
            n_clusters=self.num_clusters,
            metric='precomputed',
            linkage='average'
        )
        cluster_labels = clustering.fit_predict(distance_matrix)

        for client_id, cluster_id in enumerate(cluster_labels):
            self.client_clusters[client_id] = int(cluster_id)

        global_w = self.global_model.get_weights()
        for cluster_id in range(self.num_clusters):
            self.cluster_models[cluster_id] = create_cnn_model()
            self.cluster_models[cluster_id].set_weights(global_w)

        self.clustered = True
        return cluster_labels

    def get_model_for_client(self, client_id):
        if not self.clustered:
            return self.global_model.get_weights()
        cluster_id = self.client_clusters.get(client_id, 0)
        return self.cluster_models[cluster_id].get_weights()

    def update_cluster_models(self, client_weights_map):
        for cluster_id in range(self.num_clusters):
            cluster_cids = [cid for cid, w in client_weights_map.items() 
                           if self.client_clusters.get(cid) == cluster_id]
            if not cluster_cids:
                continue
            cluster_weights_list = [client_weights_map[cid] for cid in cluster_cids]
            avg_w = self.aggregate_weights(cluster_weights_list)
            self.cluster_models[cluster_id].set_weights(avg_w)

# --- FedCHAR Client ---
class FedCHARClient:
    def __init__(self, client_id, train_dataset, test_dataset, lambda_reg=0.01):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lambda_reg = lambda_reg
        self.personal_model = create_cnn_model()
        
    def train_proximal(self, global_weights, epochs=5):
        self.personal_model.set_weights(global_weights)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        for epoch in range(epochs):
            for x, y in self.train_dataset:
                with tf.GradientTape() as tape:
                    preds = self.personal_model(x, training=True)
                    loss_task = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, preds))
                    
                    loss_reg = 0.0
                    if self.lambda_reg > 0:
                        for w, w_global in zip(self.personal_model.trainable_variables, global_weights):
                            loss_reg += tf.reduce_sum(tf.square(w - w_global))
                        loss_reg = (self.lambda_reg / 2.0) * loss_reg
                    
                    total_loss = loss_task + loss_reg
                    
                grads = tape.gradient(total_loss, self.personal_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.personal_model.trainable_variables))
        return self.personal_model.get_weights()

    def fine_tune(self, weights, epochs=3):
        self.personal_model.set_weights(weights)
        self.personal_model.fit(self.train_dataset, epochs=epochs, verbose=0)
        return self.personal_model.get_weights()

    def compute_update(self, base_weights):
        self.personal_model.set_weights(base_weights)
        self.personal_model.fit(self.train_dataset, epochs=1, verbose=0)
        new_w = self.personal_model.get_weights()
        return [n - o for n, o in zip(new_w, base_weights)]

    def evaluate(self):
        res = self.personal_model.evaluate(self.test_dataset, verbose=0)
        return {'accuracy': res[1]}

# --- Main Run ---
def run_fedchar(num_clients=30, num_rounds=40, initial_rounds=10,alpha=0.1,
                clients_per_round=10, epochs_per_round=5, seed=42):
    
    print("="*60)
    print(f"FedCHAR (UCI HAR) | Seed: {seed}")
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
    
    server = FedCHARServer(num_clients=num_clients, num_clusters=3)
    clients = {i: FedCHARClient(i, train_datasets[i], test_datasets[i]) for i in range(num_clients)}
    metrics = FLMetricsTracker(server.global_model, train_datasets, epochs_per_round=epochs_per_round)
    
    # --- Phase 1: Warmup ---
    print(f"\n--- Phase 1: Warmup ({initial_rounds} rounds) ---")
    for round_num in range(initial_rounds):
        metrics.start_round()
        global_w = server.global_model.get_weights()
        selected = np.random.choice(list(clients.keys()), size=clients_per_round, replace=False)
        
        client_weights_list = []
        for cid in selected:
            clients[cid].lambda_reg = 0.0 
            w = clients[cid].train_proximal(global_w, epochs=epochs_per_round)
            client_weights_list.append(w)
            
        avg_w = server.aggregate_weights(client_weights_list)
        server.global_model.set_weights(avg_w)
        metrics.end_round(len(selected), sparsity=0.0)
        print(f"Round {round_num+1}/{initial_rounds} complete.")

    # --- Phase 2: Clustering ---
    print(f"\n--- Phase 2: Clustering ---")
    metrics.start_round()
    global_w = server.global_model.get_weights()
    updates = []
    all_cids = list(clients.keys())
    for cid in all_cids:
        u = clients[cid].compute_update(global_w)
        updates.append(u)
        
    server.cluster_clients(updates)
    metrics.end_round(len(selected), sparsity=0.0)
    print("Clustering Complete.")
    
    # --- Phase 3: Personalized Training ---
    print(f"\n--- Phase 3: Personalized Training ({num_rounds - initial_rounds} rounds) ---")
    for round_num in range(initial_rounds, num_rounds):
        metrics.start_round()
        selected = np.random.choice(list(clients.keys()), size=clients_per_round, replace=False)
        
        client_weights_map = {}
        for cid in selected:
            cluster_w = server.get_model_for_client(cid)
            clients[cid].lambda_reg = 0.01
            w_new = clients[cid].train_proximal(cluster_w, epochs=epochs_per_round)
            client_weights_map[cid] = w_new
            
        server.update_cluster_models(client_weights_map)
        metrics.end_round(len(selected), sparsity=0.0)
        if (round_num + 1) % 10 == 0:
            print(f"Round {round_num + 1}/{num_rounds} complete.")

    # --- Phase 4: Final Fine-Tuning ---
    print(f"\n--- Phase 4: Final Fine-Tuning ---")
    results = {}
    for cid, client in clients.items():
        cluster_w = server.get_model_for_client(cid)
        client.fine_tune(cluster_w, epochs=3)
        res = client.evaluate()
        results[cid] = res
        
    accs = [r['accuracy'] for r in results.values()]
    print(f"Avg Accuracy: {np.mean(accs):.4f}")
    
    # --- FIXED: Robust Metric Extraction ---
    raw_metrics = metrics.get_results()
    final_metrics_data = {}
    
    if isinstance(raw_metrics, list) and len(raw_metrics) > 0:
         # It's a list (unlikely based on your metrics.py, but safe to keep)
         final_metrics_data = raw_metrics[-1]
    elif isinstance(raw_metrics, dict):
         # Check if it is a history dict (keys are ints) or a flat dict (keys are strings)
         # If any value is a dict, it's a history dict.
         if any(isinstance(v, dict) for v in raw_metrics.values()):
             # Find max key (round number)
             try:
                 max_key = max(k for k in raw_metrics.keys() if isinstance(k, (int, float)))
                 final_metrics_data = raw_metrics[max_key]
             except ValueError:
                 final_metrics_data = raw_metrics
         else:
             # It is ALREADY the flat data dict
             final_metrics_data = raw_metrics

    # Calculate Per-Cluster Accuracy
    cluster_accuracies = {0: [], 1: [], 2: []}
    for cid, res in results.items():
        c_id = server.client_clusters.get(cid, -1)
        if c_id in cluster_accuracies:
            cluster_accuracies[c_id].append(res['accuracy'])

    rich_metrics = {
        'method': 'FedCHAR-UCI',
        'seed': seed,
        'run_id': f"fedchar_uci_{seed}",
        'avg_accuracy': np.mean(accs),
        'std_dev': np.std(accs),
        'min_accuracy': np.min(accs),
        'max_accuracy': np.max(accs),
        'avg_sparsity': 0.0,
        
        # Use .get safely now that final_metrics_data is guaranteed to be a dict (or empty)
        'total_comm_mb': final_metrics_data.get('total_comm_mb', 0), 
        'total_gflops': final_metrics_data.get('total_gflops', 0),
        'wall_time': final_metrics_data.get('total_training_time', 0),
        
        'acc_per_mb': np.mean(accs) / final_metrics_data.get('total_comm_mb', 1) if final_metrics_data.get('total_comm_mb', 0) > 0 else 0,
        
        'cluster_0_acc': np.mean(cluster_accuracies[0]) if cluster_accuracies[0] else 0,
        'cluster_1_acc': np.mean(cluster_accuracies[1]) if cluster_accuracies[1] else 0,
        'cluster_2_acc': np.mean(cluster_accuracies[2]) if cluster_accuracies[2] else 0,
        'cluster_assignments': server.client_clusters
    }
    
    return rich_metrics