"""
Trajectory feature storage for HiDE-Prompt continual learning.

This module stores trajectory features (hidden states) for successful trajectories
from each task, which are used for contrastive regularization.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np


class TrajectoryFeatureStorage:
    """
    Storage for trajectory features from successful episodes.

    For each task, we store:
    - Mean features of successful trajectories
    - Covariance (or variance) for sampling during task-adaptive prediction

    This is adapted from HiDE-Prompt's class centroid storage for the RL setting.
    """

    def __init__(
        self,
        storage_method: str = 'variance',  # 'variance', 'covariance', or 'multi-centroid'
        n_centroids: int = 10,  # For multi-centroid method
    ):
        """
        Initialize trajectory feature storage.

        Args:
            storage_method: Method for storing features
                - 'variance': Store mean and diagonal variance
                - 'covariance': Store mean and full covariance matrix
                - 'multi-centroid': Store multiple centroids per task using k-means
            n_centroids: Number of centroids for multi-centroid method
        """
        self.storage_method = storage_method
        self.n_centroids = n_centroids

        # Storage: task_id -> {'mean': tensor, 'cov': tensor}
        # For multi-centroid: task_id -> {'means': list of tensors, 'covs': list of tensors}
        self.task_features: Dict[int, Dict[str, any]] = {}

    def add_task_features(
        self,
        task_id: int,
        features: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Add features for a task.

        Args:
            task_id: Task ID
            features: Feature tensor of shape (num_samples, feature_dim)
            device: Device to store features on
        """
        if device is None:
            device = features.device

        features = features.to(device)

        if self.storage_method == 'variance':
            # Store mean and diagonal variance
            mean = features.mean(dim=0)  # (feature_dim,)
            # Compute covariance matrix then extract diagonal
            cov_matrix = torch.cov(features.T) + torch.eye(mean.shape[0], device=device) * 1e-4
            variance = torch.diag(cov_matrix)  # (feature_dim,)

            self.task_features[task_id] = {
                'mean': mean,
                'cov': variance,
            }

        elif self.storage_method == 'covariance':
            # Store mean and full covariance matrix
            mean = features.mean(dim=0)  # (feature_dim,)
            cov = torch.cov(features.T) + torch.eye(mean.shape[0], device=device) * 1e-4

            self.task_features[task_id] = {
                'mean': mean,
                'cov': cov,
            }

        elif self.storage_method == 'multi-centroid':
            # Use k-means to find multiple centroids
            from sklearn.cluster import KMeans

            features_np = features.cpu().numpy()
            kmeans = KMeans(n_clusters=self.n_centroids, random_state=42)
            kmeans.fit(features_np)
            cluster_labels = kmeans.labels_

            cluster_means = []
            cluster_vars = []

            for i in range(self.n_centroids):
                cluster_data = features_np[cluster_labels == i]
                if len(cluster_data) > 0:
                    cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float32, device=device)
                    cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float32, device=device)
                    cluster_means.append(cluster_mean)
                    cluster_vars.append(cluster_var)

            self.task_features[task_id] = {
                'means': cluster_means,
                'covs': cluster_vars,
            }

        else:
            raise ValueError(f"Unknown storage_method: {self.storage_method}")

    def get_task_features(self, task_id: int) -> Optional[Dict[str, any]]:
        """Get stored features for a task."""
        return self.task_features.get(task_id, None)

    def get_all_task_ids(self) -> List[int]:
        """Get all stored task IDs."""
        return list(self.task_features.keys())

    def sample_features(
        self,
        task_id: int,
        num_samples: int,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """
        Sample features from stored distribution for a task.

        Args:
            task_id: Task ID
            num_samples: Number of samples to generate
            device: Device to place samples on

        Returns:
            Sampled features of shape (num_samples, feature_dim) or None if task not found
        """
        if task_id not in self.task_features:
            return None

        task_data = self.task_features[task_id]

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.storage_method in ['variance', 'covariance']:
            mean = task_data['mean'].to(device)
            cov = task_data['cov'].to(device)

            if self.storage_method == 'variance':
                # Convert diagonal variance to full covariance matrix
                cov = torch.diag(cov)

            # Sample from multivariate normal
            from torch.distributions.multivariate_normal import MultivariateNormal
            dist = MultivariateNormal(mean.float(), cov.float())
            samples = dist.sample((num_samples,))

            return samples

        elif self.storage_method == 'multi-centroid':
            means = task_data['means']
            vars = task_data['covs']

            all_samples = []
            samples_per_centroid = num_samples // len(means)

            for mean, var in zip(means, vars):
                mean = mean.to(device)
                var = var.to(device)

                if var.mean() == 0:
                    continue

                # Sample from multivariate normal with diagonal covariance
                cov = torch.diag(var) + 1e-4 * torch.eye(mean.shape[0], device=device)
                from torch.distributions.multivariate_normal import MultivariateNormal
                dist = MultivariateNormal(mean.float(), cov.float())
                samples = dist.sample((samples_per_centroid,))
                all_samples.append(samples)

            if all_samples:
                return torch.cat(all_samples, dim=0)
            else:
                return None

    def get_all_means(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get all stored means as a single tensor.

        Returns:
            Tensor of shape (num_tasks, feature_dim) for single-centroid methods
            or (num_tasks * n_centroids, feature_dim) for multi-centroid method
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        all_means = []

        for task_id in sorted(self.task_features.keys()):
            task_data = self.task_features[task_id]

            if self.storage_method in ['variance', 'covariance']:
                all_means.append(task_data['mean'].to(device))
            elif self.storage_method == 'multi-centroid':
                for mean in task_data['means']:
                    all_means.append(mean.to(device))

        if all_means:
            return torch.stack(all_means, dim=0)
        else:
            return torch.empty(0, device=device)

    def clear(self) -> None:
        """Clear all stored features."""
        self.task_features.clear()

    def save(self, path: str) -> None:
        """Save storage to file."""
        torch.save({
            'storage_method': self.storage_method,
            'n_centroids': self.n_centroids,
            'task_features': self.task_features,
        }, path)

    def load(self, path: str) -> None:
        """Load storage from file."""
        data = torch.load(path, map_location='cpu')
        self.storage_method = data['storage_method']
        self.n_centroids = data['n_centroids']
        self.task_features = data['task_features']


def compute_contrastive_loss(
    current_features: torch.Tensor,
    stored_means: Optional[torch.Tensor],
    temperature: float = 0.8,
    reg_weight: float = 0.1,
) -> torch.Tensor:
    """
    Compute contrastive regularization loss.

    This encourages the current batch features to be orthogonal to stored
    features from previous tasks, preventing catastrophic forgetting.

    Adapted from HiDE-Prompt's orth_loss function.

    Args:
        current_features: Current batch features of shape (batch_size, feature_dim)
        stored_means: Stored mean features from previous tasks of shape (num_stored, feature_dim)
        temperature: Temperature for softmax (default: 0.8)
        reg_weight: Weight for the regularization loss (default: 0.1)

    Returns:
        Contrastive loss scalar
    """
    device = current_features.device

    if stored_means is None or stored_means.shape[0] == 0:
        # No stored features yet, use self-contrastive loss
        sim = torch.matmul(current_features, current_features.t()) / temperature
        labels = torch.arange(sim.shape[0], device=device).long()
        loss = torch.nn.functional.cross_entropy(sim, labels)
        return reg_weight * loss

    # Move stored_means to the same device as current_features
    stored_means = stored_means.to(device)

    # Combine stored means with current features
    M = torch.cat([stored_means, current_features], dim=0)  # (num_stored + batch_size, feature_dim)

    # Compute similarity matrix
    sim = torch.matmul(M, M.t()) / temperature  # (num_stored + batch_size, num_stored + batch_size)

    # Labels: each sample should be most similar to itself
    labels = torch.arange(sim.shape[0], device=device).long()

    # Cross-entropy loss
    loss = torch.nn.functional.cross_entropy(sim, labels)

    return reg_weight * loss
