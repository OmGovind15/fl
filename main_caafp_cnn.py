#
# FILENAME: main_caafp_cnn_uci.py
#
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from metrics import FLMetricsTracker
import tensorflow as tf
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import sys
from datetime import datetime
import pandas as pd

# --- CHANGED: Imported create_natural_user_split instead of non_iid ---
from data_loader import load_wisdm_dataset, create_natural_user_split, create_tf_datasets
from models_cnn import create_cnn_model, get_model_sparsity

# --- Helper: Get Trainable Layers by Order ---
def get_prunable_layers(model):
    """Returns a list of layers that have weights (kernels) to prune."""
    prunable = []
    for l in model.layers:
        # Check standard keras kernel attribute
        if hasattr(l, 'kernel') and l.kernel is not None:
            prunable.append(l)
        # Check if it has trainable weights that look like kernels (dims > 1)
        elif len(l.trainable_weights) > 0 and len(l.trainable_weights[0].shape) > 1:
            prunable.append(l)
    return prunable

# --- Logger ---
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

# --- ClusterAwareImportanceScorer ---
class ClusterAwareImportanceScorer:
    def __init__(self, alpha=0.25, beta=0.25, gamma=0.5):
        self.alpha = alpha; self.beta = beta; self.gamma = gamma

    def calculate_scores(self, dense_cluster_model, client_models_in_cluster, client_gradients_in_cluster):
        importance_scores = {}
        
        # Identify layers by INDEX to avoid name mismatch
        cluster_layers = get_prunable_layers(dense_cluster_model)
        
        for idx, layer in enumerate(cluster_layers):
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                cluster_weights = layer.kernel.numpy()
            else:
                cluster_weights = layer.trainable_weights[0].numpy()
            
            # 1. Magnitude
            magnitude_scores = np.abs(cluster_weights)
            mag_max = np.max(magnitude_scores)
            magnitude_scores = magnitude_scores / mag_max if mag_max > 0 else magnitude_scores
            
            # 2. Coherence
            client_weights_list = []
            for client_model in client_models_in_cluster:
                c_layers = get_prunable_layers(client_model)
                if idx < len(c_layers):
                    l = c_layers[idx]
                    if hasattr(l, 'kernel'): w = l.kernel.numpy()
                    else: w = l.trainable_weights[0].numpy()
                    
                    # --- FIX: Ensure shapes match before appending ---
                    if w.shape == cluster_weights.shape:
                        client_weights_list.append(w)
            
            if client_weights_list:
                variance = np.var(np.array(client_weights_list), axis=0)
                coherence_scores = 1.0 / (1.0 + variance)
                coh_max = np.max(coherence_scores)
                coherence_scores = coherence_scores / coh_max if coh_max > 0 else coherence_scores
            else:
                coherence_scores = np.ones_like(cluster_weights)
            
            # 3. Consistency
            client_grads_list = []
            for client_grads_list_ordered in client_gradients_in_cluster:
                if idx < len(client_grads_list_ordered):
                    g = client_grads_list_ordered[idx]
                    # --- FIX: Check for None and Shape match ---
                    if g is not None and g.shape == cluster_weights.shape:
                        client_grads_list.append(g)
            
            if client_grads_list:
                client_signs = np.sign(np.array(client_grads_list))
                consistency_scores = np.abs(np.mean(client_signs, axis=0))
            else:
                consistency_scores = np.ones_like(cluster_weights)
                
            # Combine
            hybrid_score = (self.alpha * magnitude_scores + 
                          self.beta * coherence_scores + 
                          self.gamma * consistency_scores)
            importance_scores[idx] = hybrid_score
            
        return importance_scores

# --- AdaptivePruningScheduler ---
class AdaptivePruningScheduler:
    def __init__(self, base_sparsity=0.7, max_sparsity=0.9, min_sparsity=0.5):
        self.base_sparsity = base_sparsity
    
    def get_sparsity_for_cluster(self, client_entropy_scores, num_clients_in_cluster):
        return 0.7

# --- CAAFPServer (CNN) ---
class CAAFPServer:
    def __init__(self, num_clients, num_clusters=3):
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.global_model = create_cnn_model()
        self.cluster_models = {}
        self.client_clusters = {}
        self.clustered = False
        
        # Stores masks by LAYER INDEX (0, 1, 2...)
        self.cluster_masks = {} 
        self.cluster_target_sparsity = {}

        self.importance_scorer = ClusterAwareImportanceScorer()
        self.pruning_scheduler = AdaptivePruningScheduler()
    
    def aggregate_weights(self, client_weights):
        avg_weights = [np.zeros_like(w) for w in client_weights[0]]
        for weights in client_weights:
            for i, w in enumerate(weights): avg_weights[i] += w
        for i in range(len(avg_weights)): avg_weights[i] /= len(client_weights)
        return avg_weights
    
    def cluster_clients(self, client_updates):
        flattened_updates = [np.concatenate([u.flatten() for u in update]) for update in client_updates]
        flattened_updates = np.array(flattened_updates)
        n_clients = len(flattened_updates)
        similarity_matrix = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(n_clients):
                similarity_matrix[i, j] = 1 - cosine(flattened_updates[i], flattened_updates[j]) if i != j else 1.0
        distance_matrix = 1 - similarity_matrix
        
        try:
            clustering = AgglomerativeClustering(n_clusters=self.num_clusters, metric='precomputed', linkage='average')
        except TypeError:
            clustering = AgglomerativeClustering(n_clusters=self.num_clusters, affinity='precomputed', linkage='average')
            
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        for client_id, cluster_id in enumerate(cluster_labels):
            self.client_clusters[client_id] = int(cluster_id)
        
        for cluster_id in range(self.num_clusters):
            self.cluster_models[cluster_id] = create_cnn_model()
            self.cluster_models[cluster_id].set_weights(self.global_model.get_weights())
            
            self.cluster_masks[cluster_id] = {}
            prunable = get_prunable_layers(self.cluster_models[cluster_id])
            
            # --- DEBUG: Layers Found ---
            print(f"[DEBUG] Cluster {cluster_id} Init: Found {len(prunable)} prunable layers.")
            
            for idx, layer in enumerate(prunable):
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    self.cluster_masks[cluster_id][idx] = np.ones_like(layer.kernel.numpy())
                else:
                    self.cluster_masks[cluster_id][idx] = np.ones_like(layer.trainable_weights[0].numpy())
        
        self.clustered = True
        return cluster_labels

    def set_cluster_sparsity_targets(self, clients):
        print("\n--- PHASE 4: Calculating Adaptive Sparsity Targets ---")
        for cluster_id in range(self.num_clusters):
            cluster_client_ids = [cid for cid, cl in self.client_clusters.items() if cl == cluster_id]
            if not cluster_client_ids:
                self.cluster_target_sparsity[cluster_id] = self.pruning_scheduler.base_sparsity
                continue
            client_entropies = [clients[cid].get_local_label_entropy() for cid in cluster_client_ids]
            target_sparsity = self.pruning_scheduler.get_sparsity_for_cluster(
                client_entropies, len(cluster_client_ids)
            )
            self.cluster_target_sparsity[cluster_id] = target_sparsity
            print(f"  Cluster {cluster_id}: Target Sparsity = {target_sparsity:.2%}")

    def update_masks_and_evolve(self, cluster_id, client_models_in_cluster, client_gradients_in_cluster, 
                               current_sparsity, target_sparsity, round_num, 
                               prune_rate=0.05, start_sparsity=0.0):
        
        print(f"  Evolving Mask for Cluster {cluster_id} (Current: {current_sparsity:.2%}, Start: {start_sparsity:.2%}, Target: {target_sparsity:.2%})")
        dense_model = self.cluster_models[cluster_id]
        
        # Calculate scores
        importance_scores = self.importance_scorer.calculate_scores(
            dense_model, client_models_in_cluster, client_gradients_in_cluster
        )
        
        avg_gradients = {}
        for idx in importance_scores.keys():
            # --- FIX: Strict Shape and None Checking ---
            # Ensure we only aggregate gradients that match the server's mask shape for this layer
            expected_shape = self.cluster_masks[cluster_id][idx].shape
            
            grads_list = []
            for c_grads in client_gradients_in_cluster:
                # Check bounds, None, and Shape
                if idx < len(c_grads):
                    g = c_grads[idx]
                    if g is not None and g.shape == expected_shape:
                        grads_list.append(g)

            if grads_list:
                avg_gradients[idx] = np.mean(np.array(grads_list), axis=0)
            else:
                avg_gradients[idx] = np.zeros_like(importance_scores[idx])
            # -------------------------------------------

        prunable_layers = get_prunable_layers(dense_model)
        
        for idx, layer in enumerate(prunable_layers):
            if idx not in importance_scores: continue
            n_deficit_push = 0
            n_churn = 0
            current_mask = self.cluster_masks[cluster_id][idx]
            hybrid_scores = importance_scores[idx]
            grad_scores = np.abs(avg_gradients[idx])
            
            n_total = current_mask.size
            n_active = np.sum(current_mask)
            
            # Recalculate local sparsity to be precise
            local_sparsity = (n_total - n_active) / n_total
            
            # --- OFFSET & PRUNING LOGIC ---
            rounds_elapsed = round_num - 10 
            pruning_steps_elapsed = rounds_elapsed // 5
            pruning_steps_total = 30 // 5
            pruning_steps_remaining = max(1, pruning_steps_total - pruning_steps_elapsed) 
            
            dist_to_start = max(0, start_sparsity - local_sparsity)
            
            if dist_to_start > 0.01: 
                n_to_prune = int(np.round(dist_to_start * n_total))
                n_to_grow = 0 # <--- CRITICAL FIX: Do not grow back immediately
                if idx == 0:
                    print(f"    [LOGIC] Initial Hard Cut. Removing {n_to_prune} weights to reach start sparsity.")

            # CASE B: EVOLUTION (Maintenance + Progressive Push)
            else:
                rounds_elapsed = round_num - 10 
                pruning_steps_elapsed = rounds_elapsed // 5
                pruning_steps_total = 30 // 5
                pruning_steps_remaining = max(1, pruning_steps_total - pruning_steps_elapsed) 
                
                sparsity_gap = target_sparsity - local_sparsity
                n_deficit_push = int(np.round(sparsity_gap * n_total / pruning_steps_remaining))
                
                # Standard churn calculation
                n_churn = int(np.round(n_active * prune_rate))
                
                if n_deficit_push > 0:
                    n_to_prune = n_churn + n_deficit_push
                    n_to_grow = n_churn
                else:
                    n_to_prune = n_churn
                    # Clamp n_deficit_push (it's negative here) to not over-grow
                    n_to_grow = n_churn - n_deficit_push
                
                if idx == 0:
                    print(f"    [LOGIC] Evolution. Deficit: {n_deficit_push}, Churn: {n_churn}. Plan: Prune {n_to_prune}, Grow {n_to_grow}")
            
            # --- SAFETY CLAMPS (The Fix) ---
            n_to_prune = int(min(n_active, max(0, n_to_prune)))
            # Cannot grow more than available zeros
            n_zeros = n_total - n_active
            n_to_grow = int(min(n_zeros, max(0, n_to_grow)))
            
            # [DEBUG] Print logic decision
            if idx == 0:
                print(f"    [LOGIC] Deficit: {n_deficit_push}, Churn: {n_churn}. Plan: Prune {n_to_prune}, Grow {n_to_grow}")

            # --- EXECUTE ---
            flat_mask = current_mask.flatten()
            flat_hybrid_scores = hybrid_scores.flatten()
            flat_grad_scores = grad_scores.flatten()

            active_indices = np.where(flat_mask > 0)[0]
            if n_to_prune > 0:
                active_scores = flat_hybrid_scores[active_indices]
                # Prune lowest scores
                indices_to_prune = active_indices[np.argsort(active_scores)[:n_to_prune]]
                flat_mask[indices_to_prune] = 0.0
            
            inactive_indices = np.where(flat_mask == 0)[0]
            if n_to_grow > 0 and inactive_indices.size > 0:
                inactive_grad_scores = flat_grad_scores[inactive_indices]
                # Grow highest gradients
                indices_to_grow = inactive_indices[np.argsort(inactive_grad_scores)[-n_to_grow:]]
                flat_mask[indices_to_grow] = 1.0
            
            self.cluster_masks[cluster_id][idx] = flat_mask.reshape(current_mask.shape)
            
            # --- DEBUG PROBE ---
            if idx == 0:
                zeros = np.sum(flat_mask == 0)
                print(f"[DIAGNOSTIC] Cluster {cluster_id} Layer 0 Post-Update: {zeros}/{n_total} zeros ({zeros/n_total:.2%}).")

    def get_final_pruned_model(self, cluster_id):
        dense_model = self.cluster_models[cluster_id]
        pruned_model = create_cnn_model() 
        pruned_model.set_weights(dense_model.get_weights())
        
        prunable = get_prunable_layers(pruned_model)
        for idx, layer in enumerate(prunable):
            if idx in self.cluster_masks[cluster_id]:
                mask = self.cluster_masks[cluster_id][idx]
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    weights = layer.kernel.numpy()
                    layer.kernel.assign(weights * mask)
                else:
                    weights = layer.trainable_weights[0].numpy()
                    layer.trainable_weights[0].assign(weights * mask)
        return pruned_model


# --- CAAFPClient (CNN + Fixes) ---
class CAAFPClient:
    def __init__(self, client_id, train_dataset, test_dataset, lambda_reg=0.01):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lambda_reg = lambda_reg
        self.personal_model = create_cnn_model() 
        self.cluster_model = create_cnn_model()
    
    def apply_mask(self, cluster_mask):
        """Forces the mask using INDEX matching."""
        prunable = get_prunable_layers(self.cluster_model)
        for idx, layer in enumerate(prunable):
            if idx in cluster_mask:
                mask = cluster_mask[idx]
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    weights = layer.kernel.numpy()
                    layer.kernel.assign(weights * mask)
                else:
                    weights = layer.trainable_weights[0].numpy()
                    layer.trainable_weights[0].assign(weights * mask)

    def train_with_regularization(self, cluster_weights, cluster_mask, epochs=5):
        self.cluster_model.set_weights(cluster_weights)
        self.apply_mask(cluster_mask) 
        
        for epoch in range(epochs):
            for x, y in self.train_dataset:
                with tf.GradientTape() as tape:
                    predictions = self.cluster_model(x, training=True)
                    task_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, predictions))
                    
                    reg_loss = 0.0
                    for cw, pw in zip(self.cluster_model.trainable_weights,
                                     self.personal_model.trainable_weights):
                        reg_loss += tf.reduce_sum(tf.square(cw - pw))
                    
                    total_loss = task_loss + (self.lambda_reg / 2.0) * reg_loss
                
                gradients = tape.gradient(total_loss, self.cluster_model.trainable_weights)
                self.cluster_model.optimizer.apply_gradients(
                    zip(gradients, self.cluster_model.trainable_weights)
                )
            
            if cluster_mask:
                self.apply_mask(cluster_mask)
        
        return self.cluster_model.get_weights()

    def fine_tune(self, pruned_model_weights, epochs=3):
        self.personal_model = create_cnn_model()
        self.personal_model.set_weights(pruned_model_weights)
        self.personal_model.fit(self.train_dataset, epochs=epochs, verbose=0)

    def compute_gradients(self, model_weights):
        self.cluster_model.set_weights(model_weights)
        
        # We need an ordered list of gradients for the kernels only
        ordered_kernel_grads = []
        
        for x, y in self.train_dataset.take(1):
            with tf.GradientTape() as tape:
                predictions = self.cluster_model(x, training=False)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, predictions))
            
            trainable_vars = self.cluster_model.trainable_variables
            grads = tape.gradient(loss, trainable_vars)
            
            # Map vars to gradients
            var_grad_map = {v.name: g for v, g in zip(trainable_vars, grads)}
            
            prunable = get_prunable_layers(self.cluster_model)
            for layer in prunable:
                target_var = None
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    target_var = layer.kernel
                elif len(layer.trainable_weights) > 0:
                    target_var = layer.trainable_weights[0]
                
                if target_var is not None and target_var.name in var_grad_map and var_grad_map[target_var.name] is not None:
                    ordered_kernel_grads.append(var_grad_map[target_var.name].numpy())
                else:
                    ordered_kernel_grads.append(None)
                    
        return ordered_kernel_grads
    
    def compute_update(self, old_weights):
        new_weights = self.cluster_model.get_weights()
        return [new - old for new, old in zip(new_weights, old_weights)]
    
    def evaluate(self):
        results = self.personal_model.evaluate(self.test_dataset, verbose=0)
        return {'loss': results[0], 'accuracy': results[1]}
    
    def get_local_label_entropy(self):
        all_labels = []
        for _, y in self.train_dataset:
            all_labels.append(y.numpy())
        if not all_labels: return 0.0
        _, counts = np.unique(np.concatenate(all_labels), return_counts=True)
        return entropy(counts / counts.sum())

# --- Main Run ---
def run_caafp(num_clients=30, num_rounds=50, initial_rounds=0,
              clustering_training_rounds=40, clients_per_round=10,
              epochs_per_round=5, fine_tune_epochs=3,
              pruning_frequency=5, prune_rate=0.05, start_sparsity=0.7,alpha=0.1,
              seed=42): # <--- ADDED SEED ARGUMENT
    
    print("="*60)
    print(f"CA-AFP (CNN): Seed {seed} | Start Sparsity: {start_sparsity:.0%}")
    print("="*60)
    
    # --- ADDED: Set Global Seeds ---
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
    
    print("Initializing server and clients...")
    server = CAAFPServer(num_clients=num_clients, num_clusters=3)
    clients = {i: CAAFPClient(i, train_datasets[i], test_datasets[i]) for i in range(num_clients)}
    metrics = FLMetricsTracker(server.global_model, train_datasets, epochs_per_round)
    
    # --- PHASE 1: Warmup ---
    print(f"\n{'='*60}\nPHASE 1: Initial Training ({initial_rounds} rounds)\n{'='*60}")
    for round_num in range(initial_rounds):
        print(f"Round {round_num + 1}/{initial_rounds}")
        metrics.start_round()
        
        if round_num < initial_rounds:
            curr_s = 0.0
        else:
            sparsities = list(server.cluster_target_sparsity.values())
            curr_s = np.mean(sparsities) if sparsities else 0.0

        selected = np.random.choice(list(clients.keys()), size=min(clients_per_round, len(clients)), replace=False)
        global_w = server.global_model.get_weights()
        
        client_weights = []
        for cid in selected:
            w = clients[cid].train_with_regularization(global_w, {}, epochs=epochs_per_round)
            client_weights.append(w)
            
        server.global_model.set_weights(server.aggregate_weights(client_weights))
        metrics.end_round(len(selected), sparsity=curr_s)
        
    # --- PHASE 2: Clustering ---
    print(f"\n{'='*60}\nPHASE 2: Client Clustering\n{'='*60}")
    metrics.start_round()
    
    global_w = server.global_model.get_weights()
    client_updates = []
    all_cids = list(clients.keys())
    
    for cid in all_cids:
        clients[cid].cluster_model.set_weights(global_w)
        clients[cid].cluster_model.fit(clients[cid].train_dataset, epochs=1, verbose=0)
        client_updates.append(clients[cid].compute_update(global_w))
        
    server.cluster_clients(client_updates)
    metrics.end_round(len(all_cids), sparsity=0.0) 
    
    print("Clustering complete!")
    server.set_cluster_sparsity_targets(clients)
    
    # --- PHASE 3: Pruning ---
    print(f"\n{'='*60}")
    print("PHASE 3: Progressive Pruning & Cluster Training (FIXED)")
    print(f"{'='*60}")
    
    total_pruning_rounds = clustering_training_rounds
    for round_num in range(initial_rounds, initial_rounds + total_pruning_rounds):
        print(f"\nRound {round_num + 1}/{initial_rounds + total_pruning_rounds}")
        metrics.start_round()
        
        is_pruning_round = (round_num - initial_rounds) % pruning_frequency == 0
        current_round_sparsities = []
        total_trained_this_round = 0 
        
        # --- FIX START: Global Selection (10 Total) ---
        # 1. Select 10 clients from the ENTIRE population first
        all_active_clients = list(clients.keys())
        n_to_pick = min(clients_per_round, len(all_active_clients))
        selected_global = np.random.choice(all_active_clients, size=n_to_pick, replace=False)
        # --- FIX END ---
        
        for cluster_id in range(server.num_clusters):
            cluster_cids = [cid for cid, cl in server.client_clusters.items() if cl == cluster_id]
            if not cluster_cids: continue
            
            # --- Pruning Step (Independent of training selection) ---
            if is_pruning_round:
                print(f"  Cluster {cluster_id}: Collecting data for mask evolution...")
                client_models = [clients[cid].personal_model for cid in cluster_cids]
                cluster_w = server.cluster_models[cluster_id].get_weights()
                client_grads = [clients[cid].compute_gradients(cluster_w) for cid in cluster_cids]
                
                curr_model = server.get_final_pruned_model(cluster_id)
                curr_sparsity = get_model_sparsity(curr_model)
                target_s = server.cluster_target_sparsity[cluster_id]
                
                server.update_masks_and_evolve(
                    cluster_id, client_models, client_grads,
                    curr_sparsity, target_s, round_num, 
                    prune_rate=prune_rate, 
                    start_sparsity=start_sparsity
                )

            # --- FIX START: Filter Global Selection ---
            # 2. Identify which of the 10 global clients belong to THIS cluster
            selected_for_cluster = [cid for cid in selected_global if cid in cluster_cids]
            
            if len(selected_for_cluster) == 0:
                # Still record sparsity for the metric tracker
                curr_model = server.get_final_pruned_model(cluster_id)
                current_round_sparsities.append(get_model_sparsity(curr_model))
                continue
                
            total_trained_this_round += len(selected_for_cluster)
            # --- FIX END ---
            
            cluster_w = server.cluster_models[cluster_id].get_weights()
            cluster_mask = server.cluster_masks[cluster_id]
            
            client_weights = []
            # 3. Train only the selected subset
            for cid in selected_for_cluster:
                w = clients[cid].train_with_regularization(cluster_w, cluster_mask, epochs=epochs_per_round)
                client_weights.append(w)
            
            avg_w = server.aggregate_weights(client_weights)
            server.cluster_models[cluster_id].set_weights(avg_w)
            
            # Enforce mask on server side aggregation
            prunable = get_prunable_layers(server.cluster_models[cluster_id])
            for idx, layer in enumerate(prunable):
                if idx in cluster_mask:
                    layer.kernel.assign(layer.kernel.numpy() * cluster_mask[idx])
            
            curr_model = server.get_final_pruned_model(cluster_id)
            current_round_sparsities.append(get_model_sparsity(curr_model))
            
            # Optional Debug
            # print(f"[DIAGNOSTIC] Cluster {cluster_id} Model Sparsity: {get_model_sparsity(server.cluster_models[cluster_id]):.2%}")

        avg_s = np.mean(current_round_sparsities) if current_round_sparsities else 0.0
        
        # Determine if mask changed this round for correct metric accounting
        mask_status = 'dynamic' if is_pruning_round else 'none'
        
        # Use new metrics.end_round signature if you updated metrics.py
        try:
            metrics.end_round(total_trained_this_round, sparsity=avg_s, mask_update_type=mask_status)
        except TypeError:
            # Fallback for old metrics.py
            metrics.end_round(total_trained_this_round, sparsity=avg_s)
    
    # --- PHASE 5: Evaluation ---
    print(f"\n{'='*60}")
    print("PHASE 5: Final Fine-Tuning & Evaluation")
    print(f"{'='*60}")
    
    results = {}
    server.final_pruned_models = {}

    for cluster_id in range(server.num_clusters):
        cluster_cids = [cid for cid, cl in server.client_clusters.items() if cl == cluster_id]
        if not cluster_cids: continue
        
        final_model = server.get_final_pruned_model(cluster_id)
        server.final_pruned_models[cluster_id] = final_model
        
        print(f"  Fine-tuning Cluster {cluster_id} clients...")
        for cid in cluster_cids:
            clients[cid].fine_tune(final_model.get_weights(), epochs=fine_tune_epochs)
            
    for cid, client in clients.items():
        res = client.evaluate()
        results[cid] = res
        
    accs = [r['accuracy'] for r in results.values()]
    final_metrics = metrics.get_results()
    print(f"\nAvg Accuracy: {np.mean(accs):.4f}")
    
    return server, clients, results, final_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # We still allow setting start_sparsity from CLI, but seeds are hardcoded list
    parser.add_argument('--start_sparsity', type=float, default=0.7)
    args = parser.parse_args()
    
    seeds = [42, 123, 456, 789, 1024]
    
    all_final_accuracies = []

    for seed in seeds:
        print(f"\n\n{'#'*60}")
        print(f"RUNNING EXPERIMENT WITH SEED: {seed}")
        print(f"{'#'*60}\n")
        
        # Capture the results to print a summary at the end
        server, clients, results, metrics = run_caafp(
            num_rounds=50, 
            prune_rate=0.05, 
            start_sparsity=args.start_sparsity,
            seed=seed  # <--- Passing the current seed
        )
        
        # Extract average accuracy from this run
        accs = [r['accuracy'] for r in results.values()]
        avg_acc = np.mean(accs)
        all_final_accuracies.append(avg_acc)
        print(f"Seed {seed} Finished. Avg Accuracy: {avg_acc:.4f}")

    print("\n" + "="*60)
    print("MULTI-SEED EXPERIMENT SUMMARY")
    print("="*60)
    for s, acc in zip(seeds, all_final_accuracies):
        print(f"Seed {s:4d}: {acc:.4f}")
    print("-" * 30)
    print(f"Mean Accuracy: {np.mean(all_final_accuracies):.4f} +/- {np.std(all_final_accuracies):.4f}")
    print("="*60)