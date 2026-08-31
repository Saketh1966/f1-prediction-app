"""
PyTorch Neural Network Architecture with Entity Embeddings for Driver, Constructor, and Circuit.
"""

from typing import Optional, List, Dict, Tuple
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseF1Model


class F1EntityEmbeddingModule(nn.Module):
    """PyTorch NN module combining entity embeddings with continuous features."""

    def __init__(
        self,
        num_drivers: int,
        num_constructors: int,
        num_circuits: int,
        num_continuous: int,
        embedding_dim: int = 16,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_p: float = 0.20,
    ):
        super().__init__()
        self.driver_emb = nn.Embedding(num_drivers + 2, embedding_dim)
        self.constructor_emb = nn.Embedding(num_constructors + 2, embedding_dim)
        self.circuit_emb = nn.Embedding(num_circuits + 2, embedding_dim)

        total_input_dim = (embedding_dim * 3) + num_continuous

        layers = []
        in_dim = total_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        driver_idx: torch.Tensor,
        constructor_idx: torch.Tensor,
        circuit_idx: torch.Tensor,
        continuous_feats: torch.Tensor,
    ) -> torch.Tensor:
        d_emb = self.driver_emb(driver_idx)
        c_emb = self.constructor_emb(constructor_idx)
        circ_emb = self.circuit_emb(circuit_idx)

        x = torch.cat([d_emb, c_emb, circ_emb, continuous_feats], dim=1)
        out = self.network(x)
        return out.squeeze(-1)


class PyTorchEntityEmbeddingModel(BaseF1Model):
    """Scikit-learn compatible wrapper around PyTorch F1 Entity Embedding NN."""

    def __init__(
        self,
        embedding_dim: int = 16,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.20,
        lr: float = 0.001,
        epochs: int = 40,
        batch_size: int = 64,
        name: str = "PyTorchEmbeddingNN",
    ):
        super().__init__(name=name)
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.driver_to_idx: Dict[int, int] = {}
        self.constructor_to_idx: Dict[int, int] = {}
        self.circuit_to_idx: Dict[int, int] = {}
        self.scaler = StandardScaler()
        self.continuous_cols: List[str] = []
        self.net: Optional[F1EntityEmbeddingModule] = None

    def _build_categorical_mappings(self, df: pd.DataFrame) -> None:
        drivers = sorted(df["driverId"].unique())
        self.driver_to_idx = {d: i + 1 for i, d in enumerate(drivers)}

        constructors = sorted(df["constructorId"].unique())
        self.constructor_to_idx = {c: i + 1 for i, c in enumerate(constructors)}

        circuits = sorted(df["circuitId"].unique())
        self.circuit_to_idx = {c: i + 1 for i, c in enumerate(circuits)}

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> "PyTorchEntityEmbeddingModel":
        self.feature_columns = list(X.columns)
        self.continuous_cols = [
            c for c in X.columns if c not in ["driverId", "constructorId", "circuitId", "raceId", "year", "round"]
        ]

        self._build_categorical_mappings(X)

        d_idx = np.array([self.driver_to_idx.get(d, 0) for d in X["driverId"]])
        c_idx = np.array([self.constructor_to_idx.get(c, 0) for c in X["constructorId"]])
        circ_idx = np.array([self.circuit_to_idx.get(circ, 0) for circ in X.get("circuitId", pd.Series(14, index=X.index))])

        X_cont = self.scaler.fit_transform(X[self.continuous_cols].fillna(0))
        y_vals = y.values.astype(np.float32)

        # Datasets
        t_d = torch.tensor(d_idx, dtype=torch.long)
        t_c = torch.tensor(c_idx, dtype=torch.long)
        t_circ = torch.tensor(circ_idx, dtype=torch.long)
        t_cont = torch.tensor(X_cont, dtype=torch.float32)
        t_y = torch.tensor(y_vals, dtype=torch.float32)

        dataset = TensorDataset(t_d, t_c, t_circ, t_cont, t_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net = F1EntityEmbeddingModule(
            num_drivers=len(self.driver_to_idx),
            num_constructors=len(self.constructor_to_idx),
            num_circuits=len(self.circuit_to_idx),
            num_continuous=len(self.continuous_cols),
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout_p=self.dropout,
        )

        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.SmoothL1Loss()

        self.net.train()
        for epoch in range(self.epochs):
            for b_d, b_c, b_circ, b_cont, b_y in loader:
                optimizer.zero_grad()
                out = self.net(b_d, b_c, b_circ, b_cont)
                loss = criterion(out, b_y)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.net is None:
            raise ValueError("Model is not fitted.")

        self.net.eval()
        d_idx = np.array([self.driver_to_idx.get(d, 0) for d in X["driverId"]])
        c_idx = np.array([self.constructor_to_idx.get(c, 0) for c in X["constructorId"]])
        circ_idx = np.array([self.circuit_to_idx.get(circ, 0) for circ in X.get("circuitId", pd.Series(14, index=X.index))])

        X_cont = self.scaler.transform(X[self.continuous_cols].fillna(0))

        t_d = torch.tensor(d_idx, dtype=torch.long)
        t_c = torch.tensor(c_idx, dtype=torch.long)
        t_circ = torch.tensor(circ_idx, dtype=torch.long)
        t_cont = torch.tensor(X_cont, dtype=torch.float32)

        with torch.no_grad():
            preds = self.net(t_d, t_c, t_circ, t_cont).cpu().numpy()

        return np.clip(preds, 1.0, 22.0)
