"""R-SHAP: Region-Based SHAP framework for 3D point clouds."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.special import comb
from tqdm.auto import tqdm


class RegionSegmentation:
    """
    Region segmentation.
    """

    @staticmethod
    def fps_voronoi(point_cloud, n_regions):
        """
        Farthest Point Sampling + Voronoi segmentation.
        Rotation-equivariant and deterministic.

        Args:
            point_cloud: (N, 3) numpy array
            n_regions: number of regions
        Returns:
            region_assignments: (N,) array of region indices
            seed_indices: (M,) array of seed point indices
        """
        N = point_cloud.shape[0]

        # Center the point cloud
        centroid = point_cloud.mean(axis=0)
        centered_pc = point_cloud - centroid

        # Farthest Point Sampling
        seed_indices = np.zeros(n_regions, dtype=np.int32)
        distances = np.full(N, np.inf)

        # First seed: farthest from centroid
        seed_indices[0] = np.argmax(np.linalg.norm(centered_pc, axis=1))

        for i in range(1, n_regions):
            # Update distances to nearest seed
            last_seed = centered_pc[seed_indices[i-1]]
            dists_to_last = np.linalg.norm(centered_pc - last_seed, axis=1)
            distances = np.minimum(distances, dists_to_last)

            # Select farthest point from all seeds
            seed_indices[i] = np.argmax(distances)

        # Voronoi assignment: assign each point to nearest seed
        seeds = centered_pc[seed_indices]
        distances_to_seeds = cdist(centered_pc, seeds)
        region_assignments = np.argmin(distances_to_seeds, axis=1)

        return region_assignments, seed_indices

    @staticmethod
    def kmeans_clustering(point_cloud, n_regions, n_init=10):
        """
        K-Means clustering.
        """
        kmeans = KMeans(n_clusters=n_regions, n_init=n_init, random_state=42)
        region_assignments = kmeans.fit_predict(point_cloud)
        return region_assignments, None

    @staticmethod
    def spectral_clustering(point_cloud, n_regions):
        """
        Spectral clustering with distance-based affinity.
        """
        # For large point clouds, use a subset
        if point_cloud.shape[0] > 1024:
            indices = np.random.choice(point_cloud.shape[0], 1024, replace=False)
            pc_subset = point_cloud[indices]
        else:
            pc_subset = point_cloud
            indices = np.arange(len(point_cloud))

        spectral = SpectralClustering(n_clusters=n_regions, affinity='nearest_neighbors',
                                     n_neighbors=10, random_state=42)
        assignments_subset = spectral.fit_predict(pc_subset)

        # Assign remaining points to nearest cluster
        if len(indices) < len(point_cloud):
            centroids = np.array([pc_subset[assignments_subset == i].mean(axis=0)
                                 for i in range(n_regions)])
            all_dists = cdist(point_cloud, centroids)
            region_assignments = np.argmin(all_dists, axis=1)
        else:
            region_assignments = assignments_subset

        return region_assignments, None


class GeometricInterpolationReference:
    """GIR for perturbing point clouds reference mechanism per its definition."""

    def __init__(self, k_neighbors=20, bandwidth='auto'):
        self.k_neighbors = k_neighbors
        self.bandwidth = bandwidth

    def perturb(self, point_cloud, present_mask):
        N = point_cloud.shape[0]
        perturbed_pc = point_cloud.copy()
        absent_indices = np.where(~present_mask)[0]
        present_indices = np.where(present_mask)[0]

        if len(present_indices) == 0:
            return np.zeros_like(point_cloud)
        if len(absent_indices) == 0:
            return perturbed_pc

        present_points = point_cloud[present_indices]
        absent_points = point_cloud[absent_indices]
        distances = cdist(absent_points, present_points)

        if self.bandwidth == 'auto':
            k = min(10, len(present_indices))
            sorted_dists = np.sort(distances, axis=1)
            col = min(k, sorted_dists.shape[1] - 1)
            bandwidth = max(np.median(sorted_dists[:, col]), 1e-6)
        else:
            bandwidth = self.bandwidth

        for idx, i in enumerate(absent_indices):
            k = min(self.k_neighbors, len(present_indices))
            nearest_k_idx = np.argpartition(distances[idx], min(k-1, len(distances[idx])-1))[:k]
            dists = distances[idx, nearest_k_idx]
            weights = np.exp(-dists**2 / (2 * bandwidth**2))
            weights = weights / (weights.sum() + 1e-10)
            perturbed_pc[i] = (weights[:, None] * present_points[nearest_k_idx]).sum(axis=0)

        return perturbed_pc


class FixedRegionSHAP:

    def __init__(self, model, reference_mechanism='gir', clustering_method='fps',
                 n_regions=8, n_samples=1000, device='cpu', value_space='logit'):
        self.model = model
        self.reference_mechanism = reference_mechanism
        self.clustering_method = clustering_method
        self.n_regions = n_regions
        self.n_samples = n_samples
        self.device = device
        self.value_space = value_space  # 'logit' or 'prob'

        if reference_mechanism == 'gir':
            self.gir = GeometricInterpolationReference(k_neighbors=20)

    def fps_voronoi(self, point_cloud, n_regions):
        """Deterministic FPS + Voronoi."""
        N = point_cloud.shape[0]
        centroid = point_cloud.mean(axis=0)
        centered_pc = point_cloud - centroid

        seed_indices = np.zeros(n_regions, dtype=np.int32)
        distances = np.full(N, np.inf)
        seed_indices[0] = np.argmax(np.linalg.norm(centered_pc, axis=1))

        for i in range(1, n_regions):
            last_seed = centered_pc[seed_indices[i-1]]
            dists_to_last = np.linalg.norm(centered_pc - last_seed, axis=1)
            distances = np.minimum(distances, dists_to_last)
            seed_indices[i] = np.argmax(distances)

        seeds = centered_pc[seed_indices]
        distances_to_seeds = cdist(centered_pc, seeds)
        region_assignments = np.argmin(distances_to_seeds, axis=1)
        return region_assignments

    def segment_regions(self, point_cloud):
        return self.fps_voronoi(point_cloud, self.n_regions)

    def create_point_mask(self, coalition, region_assignments):
        """Convert region-level coalition to point-level mask."""
        point_mask = np.zeros(len(region_assignments), dtype=bool)
        for r in range(len(coalition)):
            if coalition[r] == 1:
                point_mask[region_assignments == r] = True
        return point_mask

    def perturb_coalition(self, point_cloud, coalition, region_assignments):
        """Apply reference mechanism to absent regions."""
        point_mask = self.create_point_mask(coalition, region_assignments)
        perturbed_pc = point_cloud.copy()

        if self.reference_mechanism == 'zero':
            # Replace absent points with zeros - strong signal
            perturbed_pc[~point_mask] = 0.0

        elif self.reference_mechanism == 'mean':
            # Replace absent points with global centroid - moderate signal
            centroid = point_cloud.mean(axis=0)
            perturbed_pc[~point_mask] = centroid

        elif self.reference_mechanism == 'gir':
            # GIR interpolation - gentle signal (original)
            perturbed_pc = self.gir.perturb(point_cloud, point_mask)

        elif self.reference_mechanism == 'noise':
            # Random noise from bounding box - strong signal
            bb_min = point_cloud.min(axis=0)
            bb_max = point_cloud.max(axis=0)
            n_absent = (~point_mask).sum()
            perturbed_pc[~point_mask] = np.random.uniform(bb_min, bb_max, (n_absent, 3))

        return perturbed_pc

    def evaluate_model(self, point_cloud, target_class):
        """Evaluate model in logit or probability space."""
        self.model.eval()
        with torch.no_grad():
            pc_tensor = torch.FloatTensor(point_cloud).unsqueeze(0).to(self.device)
            logits = self.model(pc_tensor)

            if self.value_space == 'logit':
                return logits[0, target_class].item()
            else:
                probs = F.softmax(logits, dim=1)
                return probs[0, target_class].item()

    def evaluate_model_prob(self, point_cloud, target_class):
        """Return probability for faithfulness metrics."""
        self.model.eval()
        with torch.no_grad():
            pc_tensor = torch.FloatTensor(point_cloud).unsqueeze(0).to(self.device)
            logits = self.model(pc_tensor)
            probs = F.softmax(logits, dim=1)
            return probs[0, target_class].item()

    def paired_coalition_sampling(self):
        """
        Paired sampling: for each coalition S, also include complement M\\S.
        This reduces variance of Shapley estimates.
        """
        coalitions = []
        rng = np.random.RandomState(42)

        # Boundary conditions
        coalitions.append(np.zeros(self.n_regions))
        coalitions.append(np.ones(self.n_regions))

        n_pairs = (self.n_samples - 2) // 2

        for _ in range(n_pairs):
            # Sample coalition size (avoid 0 and M)
            size = rng.randint(1, self.n_regions)
            coalition = np.zeros(self.n_regions)
            selected = rng.choice(self.n_regions, size, replace=False)
            coalition[selected] = 1
            coalitions.append(coalition)
            # Add complement
            coalitions.append(1 - coalition)

        return np.array(coalitions)

    def shapley_kernel_weight(self, coalition_size, M):
        """Shapley kernel weight (standard formula)."""
        if coalition_size == 0 or coalition_size == M:
            return 0  # Handled by efficiency constraint
        return (M - 1) / (comb(M, coalition_size) * coalition_size * (M - coalition_size))

    def solve_shapley_ols(self, coalitions, values, prediction, baseline):
        """
        Solve for Shapley values using OLS with explicit efficiency constraint.
        Includes outlier clipping for numerical stability.
        """
        M = self.n_regions

        # Filter out empty and full coalitions for regression
        mask = (coalitions.sum(axis=1) > 0) & (coalitions.sum(axis=1) < M)
        Z = coalitions[mask]
        y = values[mask]

        # Outlier clipping: remove extreme value function evaluations
        # Uses MAD-based robust clipping to handle non-linear value function responses
        median_y = np.median(y)
        mad = np.median(np.abs(y - median_y))
        if mad > 1e-10:
            clip_std = mad * 1.4826  # MAD to std conversion
            clip_lo = median_y - 5 * clip_std
            clip_hi = median_y + 5 * clip_std
            y = np.clip(y, clip_lo, clip_hi)

        # Compute Shapley kernel weights
        weights = np.array([self.shapley_kernel_weight(int(z.sum()), M) for z in Z])
        weights = weights / (weights.max() + 1e-10)  # Normalize

        # Weighted design matrix
        W = np.diag(np.sqrt(weights))
        Zw = W @ Z
        yw = W @ y

        # OLS solution: phi = (Z'WZ)^{-1} Z'Wy
        # Add small regularization only for numerical stability
        ZtWZ = Zw.T @ Zw + 1e-8 * np.eye(M)
        ZtWy = Zw.T @ yw

        phi = np.linalg.solve(ZtWZ, ZtWy)

        # Enforce efficiency constraint: sum(phi) = prediction - baseline
        expected_sum = prediction - baseline
        current_sum = phi.sum()

        if abs(current_sum) > 1e-10:
            # Project onto efficiency constraint while preserving relative magnitudes
            phi = phi + (expected_sum - current_sum) / M
        else:
            # Fallback: uniform distribution
            phi = np.full(M, expected_sum / M)

        return phi

    def single_region_occlusion(self, point_cloud, region_assignments, target_class):
        """
        Diagnostic: check if removing individual regions changes model output.
        If all drops are < 0.01, the model is insensitive to regional removal.
        """
        prediction = self.evaluate_model(point_cloud, target_class)
        drops = []

        for r in range(self.n_regions):
            coalition = np.ones(self.n_regions)
            coalition[r] = 0

            perturbed = self.perturb_coalition(point_cloud, coalition, region_assignments)
            value = self.evaluate_model(perturbed, target_class)
            drops.append(prediction - value)

        return np.array(drops)

    def explain(self, point_cloud, target_class=None, verbose=True):
        """
        R-SHAP explanation.
        """
        # Segment
        region_assignments = self.segment_regions(point_cloud)

        # Get target class
        if target_class is None:
            pc_tensor = torch.FloatTensor(point_cloud).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(pc_tensor)
                target_class = torch.argmax(logits).item()

        # Baseline (all absent) and prediction (all present)
        baseline_pc = self.perturb_coalition(
            point_cloud, np.zeros(self.n_regions), region_assignments
        )
        baseline = self.evaluate_model(baseline_pc, target_class)
        prediction = self.evaluate_model(point_cloud, target_class)

        if verbose:
            print(f"  Prediction ({self.value_space}): {prediction:.4f}")
            print(f"  Baseline   ({self.value_space}): {baseline:.4f}")
            print(f"  Delta: {prediction - baseline:.4f}")

        # Sample coalitions (paired for variance reduction)
        coalitions = self.paired_coalition_sampling()

        # Evaluate model on each coalition
        values = []
        iterator = tqdm(coalitions, desc="  Evaluating", leave=False) if verbose else coalitions

        for coalition in iterator:
            perturbed_pc = self.perturb_coalition(point_cloud, coalition, region_assignments)
            value = self.evaluate_model(perturbed_pc, target_class)
            values.append(value)

        coalitions = np.array(coalitions)
        values = np.array(values)

        # Solve with OLS + efficiency constraint (NO Ridge)
        region_importances = self.solve_shapley_ols(
            coalitions, values, prediction, baseline
        )

        return region_importances, baseline, prediction, region_assignments, target_class
