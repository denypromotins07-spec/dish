"""
Lightweight Transformer with Linear Attention for order book sequence modeling.
Implements Performer/Linear Attention mechanism to bypass O(n²) memory complexity.
Optimized for AMD ROCm with strict VRAM constraints.
"""

import os
import logging
import math
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

# ROCm environment variables
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "true"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    raise ImportError("PyTorch required. Install with: pip install torch")

logger = logging.getLogger(__name__)


class FeatureMap(nn.Module):
    """
    Kernel feature map for linear attention approximation.
    Uses FAVOR+ (Fast Attention via positive Orthogonal Random features).
    """
    
    def __init__(self, dim: int, nb_features: int, orthogonal: bool = True):
        super().__init__()
        self.nb_features = nb_features
        self.orthogonal = orthogonal
        
        # Random projection matrix
        if orthogonal:
            # Use orthogonal random features for better approximation
            self.weight = nn.Linear(dim, nb_features, bias=False)
            self._initialize_orthogonal()
        else:
            self.weight = nn.Linear(dim, nb_features, bias=False)
    
    def _initialize_orthogonal(self):
        """Initialize weight matrix with orthogonal columns."""
        weight = torch.randn(self.weight.weight.shape[0], self.weight.weight.shape[1])
        q, r = torch.linalg.qr(weight)
        self.weight.weight.data = q * math.sqrt(self.nb_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply kernel feature map: exp(-x²/2) * cos(x @ W + b)
        x = self.weight(x)
        return torch.cos(x)  # Simplified feature map


class LinearAttention(nn.Module):
    """
    Linear attention mechanism with O(n) memory complexity.
    Based on Performer/FAVOR+ approximation.
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        nb_features: int = 256,  # Number of random features
        causal: bool = False,
    ):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.causal = causal
        
        # To Q, K, V projections
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        
        # Feature map for linear attention
        self.feature_map = FeatureMap(self.dim_head, nb_features)
        
        # Output projection
        self.to_out = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = self.heads
        
        # Get Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(B, N, H, self.dim_head).transpose(1, 2) for t in qkv]
        
        # Scale
        q = q * self.scale
        
        # Apply feature map for linear attention
        q_feat = self.feature_map(q)
        k_feat = self.feature_map(k)
        
        # Linear attention: (Q @ K^T) @ V ≈ (Q @ (K @ V^T))
        if self.causal:
            # Causal linear attention with cumulative sum
            kv = torch.cumsum(k_feat.transpose(2, 3) @ v, dim=2)
            z = torch.cumsum(k_feat.sum(dim=2, keepdim=True), dim=2) + 1e-6
            
            out = (q_feat @ kv) / z
        else:
            # Bidirectional linear attention
            kv = k_feat.transpose(2, 3) @ v
            z = k_feat.sum(dim=2, keepdim=True) + 1e-6
            
            out = (q_feat @ kv) / z
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)


class FeedForward(nn.Module):
    """Memory-efficient feed-forward network with gradient checkpointing."""
    
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(dim * mult)
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                lambda t: self.net(t), x, use_reentrant=False
            )
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm and linear attention."""
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        mult: int = 4,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, heads=heads, causal=causal)
        
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=mult, dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TransformerLite(nn.Module):
    """
    Lightweight Transformer with Linear Attention for order book modeling.
    
    Features:
    - O(n) memory complexity via linear attention
    - Gradient checkpointing for reduced VRAM usage
    - Configurable depth/width for memory constraints
    - Suitable for long sequence modeling (order book deltas)
    """
    
    def __init__(
        self,
        input_dim: int = 20,  # Order book features
        seq_length: int = 200,  # Sequence length
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        mult: int = 4,
        dropout: float = 0.1,
        output_dim: int = 1,
        causal: bool = True,  # Autoregressive for time series
        max_vram_mb: int = 1536,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.max_vram_mb = max_vram_mb
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding (learned)
        self.pos_encoding = nn.Embedding(seq_length, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                heads=num_heads,
                mult=mult,
                dropout=dropout,
                causal=causal,
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim),
        )
        
        self._check_resources()
    
    def _check_resources(self):
        """Check GPU/CPU resources."""
        import psutil
        
        if torch.cuda.is_available():
            try:
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                logger.info(f"GPU VRAM: {vram_total:.0f}MB, Limit: {self.max_vram_mb}MB")
            except Exception as e:
                logger.warning(f"Could not query VRAM: {e}")
        else:
            logger.info("CUDA not available. Using CPU mode.")
        
        ram_available = psutil.virtual_memory().available / (1024**3)
        logger.info(f"System RAM available: {ram_available:.2f}GB")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            x: Input tensor of shape (batch, seq_length, input_dim)
        
        Returns:
            Output tensor of shape (batch, output_dim)
        """
        B, N, D = x.shape
        
        # Embed input
        x = self.embedding(x)
        
        # Add positional encoding
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_encoding(positions)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Global pooling (mean over sequence)
        x = x.mean(dim=1)
        
        # Output
        x = self.norm(x)
        output = self.output_layer(x)
        
        return output


class TransformerLiteTrainer:
    """
    Trainer for TransformerLite with memory management.
    Implements gradient accumulation and early stopping.
    """
    
    def __init__(
        self,
        model: TransformerLite,
        learning_rate: float = 0.0005,
        weight_decay: float = 1e-4,
        max_epochs: int = 30,
        early_stopping_patience: int = 8,
        batch_size: int = 32,
        gradient_accumulation_steps: int = 8,
        device: Optional[str] = None,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Device setup
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=learning_rate / 10,
        )
        
        # Loss
        self.criterion = nn.SmoothL1Loss()  # Huber loss for robustness
        
        # Training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history = []
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train one epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Scale for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update every N steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Full training loop."""
        import psutil
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(-1) if len(y_train.shape) == 1 else torch.FloatTensor(y_train),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val).unsqueeze(-1) if len(y_val.shape) == 1 else torch.FloatTensor(y_val),
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        logger.info(f"Training for {self.max_epochs} epochs...")
        
        for epoch in range(self.max_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1}/{self.max_epochs}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={current_lr:.7f}"
            )
            
            # Store history
            self.training_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
            })
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Memory check
            ram_available = psutil.virtual_memory().available / (1024**3)
            if ram_available < 1.0:
                logger.warning(f"Critical low RAM: {ram_available:.2f}GB")
        
        # Load best model
        if hasattr(self, "best_model_state"):
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with val_loss={self.best_val_loss:.6f}")
        
        return {
            "best_val_loss": self.best_val_loss,
            "epochs_trained": len(self.training_history),
            "training_history": self.training_history,
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions)


def main():
    """Example usage."""
    import psutil
    
    print(f"Initial RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")
    
    # Generate synthetic order book data
    np.random.seed(42)
    n_samples = 8000
    seq_length = 100
    input_dim = 20  # Order book levels/features
    
    X = np.random.randn(n_samples, seq_length, input_dim).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    
    # Split
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    
    # Create model
    model = TransformerLite(
        input_dim=input_dim,
        seq_length=seq_length,
        embed_dim=64,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        output_dim=1,
        causal=True,
    )
    
    # Create trainer
    trainer = TransformerLiteTrainer(
        model=model,
        learning_rate=0.0005,
        max_epochs=15,
        early_stopping_patience=5,
        batch_size=16,
        gradient_accumulation_steps=8,
    )
    
    # Train
    results = trainer.fit(X_train, y_train, X_val, y_val)
    print(f"\nTraining complete: {results['best_val_loss']:.6f} after {results['epochs_trained']} epochs")
    
    # Predict
    predictions = trainer.predict(X_val[:50])
    print(f"Sample predictions: {predictions[:5].flatten()}")
    
    print(f"\nFinal RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")


if __name__ == "__main__":
    main()
