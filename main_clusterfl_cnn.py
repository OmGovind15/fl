#
# FILENAME: main_clusterfl_cnn_uci.py
#
import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

# --- CHANGED: Use UCI Data Loader ---
from data_loader import load_wisdm_dataset, create_natural_user_split, create_tf_datasets
# --- CHANGED: Use UCI Model ---
from models_cnn import create_cnn_model
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

# --- ClusterFL Client ---
class ClusterFLClient:
    def __init__(self, client_id, train_dataset, test_dataset):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = create_cnn_model()
        
        # Initialize local weights
        self.weights = [w.numpy() for w in self.model.trainable_variables]
        
    def update_weights(self, lambda_i, z_i, epochs=1, lr=0.001):
        """
        Implements the ClusterFL Node Update (Equation 6 in the paper).
        Objective: min f_i(w) + lambda_i * || w - z_i / (2*lambda_i) ||^2
        """
        self.model.set_weights(self.weights)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        # Calculate the target for the proximal term: target = z_i / (2 * lambda_i)
        if lambda_i > 1e-8:
            reg_target = [z / (2.0 * lambda_i) for z in z_i]
        else:
            # If lambda is 0, no regularization
            reg_target = self.weights 

        for epoch in range(epochs):
            for x, y in self.train_dataset:
                with tf.GradientTape() as tape:
                    # 1. Empirical Loss (f_i)
                    preds = self.model(x, training=True)
                    loss_emp = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, preds))
                    
                    # 2. ClusterFL Regularization (Proximal Term)
                    loss_reg = 0.0
                    if lambda_i > 1e-8:
                        for w, t in zip(self.model.trainable_variables, reg_target):
                            # || w - target ||^2
                            loss_reg += tf.reduce_sum(tf.square(w - t))
                        # Multiply by lambda_i
                        loss_reg *= lambda_i
                    
                    total_loss = loss_emp + loss_reg
                    
                grads = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Save updated weights locally
        self.weights = self.model.get_weights()
        return self.weights

    def evaluate(self):
        self.model.set_weights(self.weights)
        res = self.model.evaluate(self.test_dataset, verbose=0)
        return {'accuracy': res[1]}

# --- ClusterFL Server ---
class ClusterFLServer:
    def __init__(self, num_clients, num_clusters=3, 
                 alpha=1e-3, beta=5e-4, rho=5e-3):
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        
        # Hyperparameters (Section 8.3 Analysis: rho/beta ratio ~10 is stable)
        self.alpha = alpha   
        self.beta = beta     
        self.rho = rho       
        
        # 1. Initialize Weights Storage
        self.global_model = create_cnn_model() 
        self.W_shapes = [w.shape for w in self.global_model.trainable_variables]
        
        # Store weights for ALL clients (needed for ADMM sum)
        # Initialize with random weights (same seed for fairness)
        init_weights = self.global_model.get_weights()
        self.client_weights = {i: [np.copy(w) for w in init_weights] for i in range(num_clients)}
        
        # 2. Initialize Cluster Indicator Matrix F (Relaxed)
        # Random initialization or K-Means initialization is common
        # Here we start with uniform assignment to allow spectral clustering to learn
        self.F = np.zeros((num_clients, num_clusters))
        for i in range(num_clients):
            self.F[i, i % num_clusters] = 1.0 # Simple cyclic init
            
        # 3. Initialize ADMM Auxiliaries (Omega, U)
        # Stored as list of numpy arrays (matching model layers) for each cluster
        self.Omega = {k: [np.zeros(s) for s in self.W_shapes] for k in range(num_clusters)}
        self.U     = {k: [np.zeros(s) for s in self.W_shapes] for k in range(num_clusters)}

    def learn_cluster_structure(self):
        """
        Updates F using Spectral Clustering on current client weights (Section 5.3).
        Uses Cosine Similarity as metric.
        """
        # Flatten weights for clustering
        flat_weights_matrix = []
        for i in range(self.num_clients):
            # Concat all layers into one vector per client
            flat = np.concatenate([w.flatten() for w in self.client_weights[i]])
            flat_weights_matrix.append(flat)
        flat_weights_matrix = np.array(flat_weights_matrix)
        
        # Compute Affinity (Cosine Similarity + 1 to make it positive)
        affinity = cosine_similarity(flat_weights_matrix) + 1.0
        
        # Spectral Clustering
        try:
            sc = SpectralClustering(n_clusters=self.num_clusters, 
                                    affinity='precomputed', 
                                    assign_labels='kmeans')
            labels = sc.fit_predict(affinity)
        except Exception as e:
            print(f"  [Warning] Spectral Clustering failed: {e}. Keeping old clusters.")
            return
        
        # Update F (Relaxed Indicator)
        # F_ik = 1/sqrt(|Ck|) if i in cluster k, else 0
        self.F = np.zeros((self.num_clients, self.num_clusters))
        for k in range(self.num_clusters):
            members = np.where(labels == k)[0]
            if len(members) > 0:
                val = 1.0 / np.sqrt(len(members))
                self.F[members, k] = val
        
        print(f"  Cluster sizes: {[len(np.where(labels==k)[0]) for k in range(self.num_clusters)]}")

    def update_admm_auxiliaries(self):
        """
        Server Step: Update Omega and U (Eq 7 & 8).
        """
        denom = self.rho - 2 * self.beta
        if abs(denom) < 1e-8: denom = 1e-8
        
        for l_idx in range(len(self.W_shapes)):
            # Gather W for layer l_idx from all clients: (M, layer_shape...)
            W_layer_stack = np.stack([self.client_weights[i][l_idx] for i in range(self.num_clients)])
            
            # Flatten layer dim for matrix mult: (M, D_layer)
            # W_flat: M x D
            # F: M x K
            layer_size = np.prod(self.W_shapes[l_idx])
            W_flat = W_layer_stack.reshape(self.num_clients, -1)
            
            # FTW = F.T @ W -> (K, D)
            FTW = self.F.T @ W_flat
            
            # Get U_flat: (K, D)
            U_flat = np.stack([self.U[k][l_idx].flatten() for k in range(self.num_clusters)])
            
            # Update Omega (Eq derived from minimizing Augmented Lagrangian)
            # Omega = (rho * FTW + U) / (rho - 2*beta)
            Omega_new_flat = (self.rho * FTW + U_flat) / denom
            
            # Update U
            # U = U + rho * (FTW - Omega)
            U_new_flat = U_flat + self.rho * (FTW - Omega_new_flat)
            
            # Store back
            for k in range(self.num_clusters):
                self.Omega[k][l_idx] = Omega_new_flat[k].reshape(self.W_shapes[l_idx])
                self.U[k][l_idx]     = U_new_flat[k].reshape(self.W_shapes[l_idx])

    def get_collaborative_vars(self, client_id):
        """
        Computes lambda_i and z_i for a specific client (Eq 6 inputs).
        """
        F_i = self.F[client_id] # (K,)
        
        # 1. Lambda_i = (rho/2) * sum(F_ik^2)
        lambda_i = (self.rho / 2.0) * np.sum(F_i**2)
        
        # 2. z_i (The aggregated target)
        # z_i = Sum_k [ F_ik * (rho*Omega_k - U_k - rho * NeighborSum_k) ]
        # NeighborSum_k = (Sum_{q!=i} F_qk * w_q)
        # This simplifies to: (F.T @ W)_k - F_ik * w_i
        
        z_i = []
        for l_idx in range(len(self.W_shapes)):
            w_i_flat = self.client_weights[client_id][l_idx].flatten()
            
            # Recompute global sum for this layer (expensive but correct)
            # Optimization: Could cache (F.T @ W) at start of round
            W_layer_stack = np.stack([self.client_weights[c][l_idx] for c in range(self.num_clients)])
            W_flat = W_layer_stack.reshape(self.num_clients, -1)
            FTW = self.F.T @ W_flat # (K, D)
            
            z_layer_acc = np.zeros_like(w_i_flat)
            
            for k in range(self.num_clusters):
                if F_i[k] == 0: continue
                
                Omega_k = self.Omega[k][l_idx].flatten()
                U_k     = self.U[k][l_idx].flatten()
                
                # Sum of neighbors in cluster k (excluding client i)
                neighbor_sum = FTW[k] - (F_i[k] * w_i_flat)
                
                # ADMM Term
                term = (self.rho * Omega_k) - U_k - (self.rho * neighbor_sum)
                
                z_layer_acc += F_i[k] * term
                
            z_i.append(z_layer_acc.reshape(self.W_shapes[l_idx]))
            
        return lambda_i, z_i

# --- Main Run ---
def run_clusterfl(num_clients=30, num_rounds=50,alpha=0.1,seed=42, clients_per_round=10):
    
    print("="*60)
    print(f"ClusterFL (CNN) for UCI HAR Dataset")
    print("="*60)
    
    tf.keras.backend.clear_session()
    
    # 1. Data Setup (UCI)
    print("\n[INFO] Loading WISDM Data...")
    X_data, y_data, user_ids = load_wisdm_dataset() # uses default path

    # Use Natural User Split (1 User = 1 Client)
    client_data = create_natural_user_split(X_data, y_data, user_ids)
    
    # Update num_clients dynamically based on actual users found
    num_clients = len(client_data)
    print(f"[INFO] Total Clients: {num_clients}")

    # Create TF Datasets (using safe per-client scaling)
    train_datasets, test_datasets = create_tf_datasets(client_data, batch_size=32)
    
    # 2. Init System
    server = ClusterFLServer(num_clients, num_clusters=3, alpha=1e-3, beta=5e-4, rho=5e-3)
    clients = {i: ClusterFLClient(i, train_datasets[i], test_datasets[i]) for i in range(num_clients)}
    
    # 3. Federated Loop
    print("\n--- Starting ClusterFL Training ---")
    metrics = FLMetricsTracker(server.global_model, train_datasets)
    run_metrics = {}

    for round_num in range(num_rounds):
        metrics.start_round()
        # A. Learn Cluster Structure (Every 5 rounds per paper heuristic)
        if round_num % 5 == 0:
            print(f"  [Round {round_num}] Updating Cluster Structure (Spectral)...")
            server.learn_cluster_structure()
            
        # B. Server updates ADMM auxiliaries (Omega, U) using *latest* client weights
        server.update_admm_auxiliaries()
        
        # C. Select Clients
        selected_clients = np.random.choice(list(clients.keys()), size=clients_per_round, replace=False)
        
        # D. Client Updates
        for cid in selected_clients:
            # 1. Server calculates collaborative vars (lambda, z) for this client
            lam, z = server.get_collaborative_vars(cid)
            
            # 2. Client updates weights solving the proximal subproblem
            # Note: We use 3 epochs to simulate sufficient local solving steps
            w_new = clients[cid].update_weights(lam, z, epochs=3)
            
            # 3. Server updates its state for this client
            server.client_weights[cid] = w_new
        
        metrics.end_round(len(selected_clients), sparsity=0.0)
        
        if (round_num + 1) % 10 == 0:
            print(f"Round {round_num + 1}/{num_rounds} complete.")

    # 4. Final Evaluation
    print("\nFinal Evaluation:")
    results = {}
    for cid, client in clients.items():
        res = client.evaluate()
        results[cid] = res
        
    accs = [r['accuracy'] for r in results.values()]
    print(f"Avg Accuracy: {np.mean(accs):.4f}")
    
    # Capture the tracked metrics
    run_metrics = metrics.get_results()
    
    return server, clients, results, run_metrics

if __name__ == "__main__":
    run_id = f"clusterfl_cnn_uci_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs('logs', exist_ok=True)
    logger = Logger(f'logs/run_log_{run_id}.log')
    sys.stdout = logger
    run_clusterfl()