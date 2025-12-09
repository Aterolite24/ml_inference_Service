import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from app.models.registry import BaseModel, ModelRegistry

@ModelRegistry.register("kmeans")
class KMeansModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.model = KMeans(**valid_kwargs)

    def train(self, X: np.ndarray, y: np.ndarray = None) -> None:
        self.model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

@ModelRegistry.register("dbscan")
class DBSCANModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.model = DBSCAN(**valid_kwargs)

    def train(self, X: np.ndarray, y: np.ndarray = None) -> None:
        # DBSCAN is transductive, it fits and predicts on the same data usually (for clustering structure)
        # However, for 'train', we will just fit_predict and store labels if needed.
        # But wait, sklearn DBSCAN doesn't verify predict on new data easily without extensions. 
        # But we can just fit.
        self.model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # DBSCAN does not implement a predict method for new data in sklearn.
        # We will return the labels_ of the fitted data if X is the same as training data,
        # otherwise we can implement a basic 1-NN classification to assign clusters to new points 
        # or re-run fit_predict (expensive).
        # For simplicity in this demo, we'll return fitted labels if X shape matches, else -1.
        # Ideally, we should use a transductive approach or HDBSCAN, but for now:
        if hasattr(self.model, 'labels_') and len(X) == len(self.model.labels_):
             return self.model.labels_
        return np.full(len(X), -1)

@ModelRegistry.register("dpc")
class DPCModel(BaseModel):
    """
    Basic implementation of Density Peak Clustering.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dc = kwargs.get('dc', 0.1) # Cutoff distance
        self.rho = None
        self.delta = None
        self.labels_ = None
        self.cluster_centers_ = None

    def train(self, X: np.ndarray, y: np.ndarray = None) -> None:
        # Simple DPC implementation
        n_samples = X.shape[0]
        # Calculate distance matrix (simplified)
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(X)
        
        # Calculate local density rho
        # using a gaussian kernel or simple cutoff
        self.rho = np.zeros(n_samples)
        for i in range(n_samples):
             # Simple cutoff kernel
             self.rho[i] = np.sum(dists[i] < self.dc) - 1 # exclude self
        
        # Calculate delta (min distance to higher density point)
        self.delta = np.zeros(n_samples)
        sorted_rho_idx = np.argsort(self.rho)[::-1]
        
        self.delta[sorted_rho_idx[0]] = np.max(dists[sorted_rho_idx[0]])
        
        for i in range(1, n_samples):
            idx = sorted_rho_idx[i]
            higher_density_indices = sorted_rho_idx[:i]
            self.delta[idx] = np.min(dists[idx][higher_density_indices])

        # Assign cluster centers (simplified heuristic: look for high rho and high delta)
        # For this demo, we'll just take top K points as centers using a threshold or K
        # Let's say we want a dynamic K or fixed?
        # Let's use a simple decision graph approach: rho * delta
        gamma = self.rho * self.delta
        # Select centers as outliers in gamma (top N)
        # Heuristic: top % or threshold. Let's say top 3 for demo.
        n_clusters = 3
        center_indices = np.argsort(gamma)[-n_clusters:]
        self.cluster_centers_ = X[center_indices]
        
        # Assign rest of points
        self.labels_ = np.zeros(n_samples, dtype=int) - 1
        for i, c_idx in enumerate(center_indices):
            self.labels_[c_idx] = i

        # Assign remaining points to same cluster as nearest neighbor with higher density
        for i in range(n_samples):
            idx = sorted_rho_idx[i]
            if self.labels_[idx] != -1:
                continue
            
            # Find nearest neighbor with higher density
            # Effectively, looks at sorted_rho_idx[:i]
            higher_density_indices = sorted_rho_idx[:i]
            nearest_neighbor_in_higher = higher_density_indices[np.argmin(dists[idx][higher_density_indices])]
            self.labels_[idx] = self.labels_[nearest_neighbor_in_higher]

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Assign new points to nearest cluster center
        if self.cluster_centers_ is None:
             return np.full(len(X), -1)
        
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(X, self.cluster_centers_)
        return np.argmin(dists, axis=1)

