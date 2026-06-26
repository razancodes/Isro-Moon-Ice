"""
Few-Shot Classification Head
==============================

Attach a lightweight classification head to frozen LunarFM embeddings
for terrain type segmentation.

Following the pattern established in the LunarFM repository's
EmbeddingClassifier (src/lunarlab/models/embedding_classifier.py), 
but adapted for few-shot scenarios with very small labeled datasets.

Supports two modes:
1. Linear probe: Simple linear layer on frozen embeddings (best for <50 examples)
2. MLP head: Multi-layer classifier (better for 50+ examples per class)
"""

import numpy as np
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


class LinearProbe(nn.Module):
    """Simple linear classifier on frozen embeddings. Best for very few shots."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)


class MLPClassificationHead(nn.Module):
    """
    Multi-layer classification head for embeddings.
    
    Architecture (following the repo's EmbeddingClassifier pattern):
    768 -> 256 -> ReLU -> Dropout -> 128 -> ReLU -> Dropout -> N_classes
    
    Lighter than the repo's 768->512->256->128->N version, suited for
    fewer training examples to reduce overfitting.
    """
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        return self.model(x)


class TerrainClassifier:
    """
    High-level wrapper for training and evaluating a terrain classifier
    on LunarFM embeddings.
    
    Usage:
        classifier = TerrainClassifier(
            num_classes=5,
            class_names=['crater_floor', 'rim', 'ejecta', 'smooth', 'rough'],
            mode='mlp',  # or 'linear'
        )
        
        classifier.fit(
            embeddings=train_embeddings,  # [N, 768]
            labels=train_labels,          # [N] integer labels
            val_fraction=0.2,
        )
        
        predictions = classifier.predict(test_embeddings)
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        input_dim: int = 768,
        mode: str = 'mlp',
        dropout: float = 0.3,
        device: str = 'cpu',
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.input_dim = input_dim
        self.mode = mode
        self.device = device
        
        if mode == 'linear':
            self.model = LinearProbe(input_dim, num_classes)
        elif mode == 'mlp':
            self.model = MLPClassificationHead(input_dim, num_classes, dropout=dropout)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'linear' or 'mlp'.")
        
        self.model = self.model.to(device)
        self.trained = False
    
    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        val_fraction: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Train the classification head.
        
        Args:
            embeddings: [N, embed_dim] array of frozen embeddings
            labels: [N] integer class labels
            val_fraction: Fraction of data for validation
            learning_rate: Initial learning rate
            epochs: Maximum training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            verbose: Print training progress
            
        Returns:
            Training history dict with 'train_loss', 'val_loss', 'val_acc' lists
        """
        # Split data
        if val_fraction > 0 and len(embeddings) > 10:
            X_train, X_val, y_train, y_val = train_test_split(
                embeddings, labels, test_size=val_fraction, 
                stratify=labels, random_state=42
            )
        else:
            X_train, y_train = embeddings, labels
            X_val, y_val = embeddings, labels  # Use train as val if too few samples
        
        # Convert to tensors
        train_X = torch.from_numpy(X_train).float().to(self.device)
        train_y = torch.from_numpy(y_train).long().to(self.device)
        val_X = torch.from_numpy(X_val).float().to(self.device)
        val_y = torch.from_numpy(y_val).long().to(self.device)
        
        # DataLoader
        train_dataset = TensorDataset(train_X, train_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        best_state = None
        no_improve = 0
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            
            # Validate
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(val_X)
                val_loss = criterion(val_logits, val_y).item()
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == val_y).float().mean().item()
            
            scheduler.step()
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                logger.info(
                    f"Epoch {epoch:3d}/{epochs}: "
                    f"train_loss={avg_train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"val_acc={val_acc:.3f}"
                )
            
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)
        
        self.trained = True
        logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
        
        return history
    
    @torch.no_grad()
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict class labels for embeddings.
        
        Args:
            embeddings: [N, embed_dim] array
            
        Returns:
            [N] integer class labels
        """
        self.model.eval()
        X = torch.from_numpy(embeddings).float().to(self.device)
        logits = self.model(X)
        return logits.argmax(dim=1).cpu().numpy()
    
    @torch.no_grad()
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for embeddings.
        
        Args:
            embeddings: [N, embed_dim] array
            
        Returns:
            [N, num_classes] probability array
        """
        self.model.eval()
        X = torch.from_numpy(embeddings).float().to(self.device)
        logits = self.model(X)
        return F.softmax(logits, dim=1).cpu().numpy()
    
    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        verbose: bool = True,
    ) -> dict:
        """
        Evaluate the classifier and print a classification report.
        
        Args:
            embeddings: [N, embed_dim] test embeddings
            labels: [N] true labels
            verbose: Print detailed report
            
        Returns:
            Dict with accuracy, per-class metrics
        """
        predictions = self.predict(embeddings)
        
        report = classification_report(
            labels, predictions,
            target_names=self.class_names,
            output_dict=True,
        )
        
        if verbose:
            logger.info("\nClassification Report:")
            print(classification_report(
                labels, predictions,
                target_names=self.class_names,
            ))
            
            logger.info("\nConfusion Matrix:")
            cm = confusion_matrix(labels, predictions)
            print(cm)
        
        return report
    
    def save(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state': self.model.state_dict(),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'input_dim': self.input_dim,
            'mode': self.mode,
        }, path)
        logger.info(f"Saved classifier to {path}")
    
    def load(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model = self.model.to(self.device)
        self.trained = True
        logger.info(f"Loaded classifier from {path}")
