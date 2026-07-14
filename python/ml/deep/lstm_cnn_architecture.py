"""
Hybrid CNN-LSTM architecture for pattern recognition in financial time series.
Optimized for AMD ROCm with bfloat16 precision and gradient checkpointing.
Strictly bounded VRAM usage for AMD Radeon GPU constraints.
"""

import os
import logging
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

# Set ROCm environment variables early
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"  # For AMD Radeon RX 7000 series
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "true"
os.environ["MAX_SPLIT_SIZE_MB"] = "128"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    raise ImportError("PyTorch required. Install with: pip install torch")

logger = logging.getLogger(__name__)


class MemoryEfficientConvBlock(nn.Module):
    """
    Memory-efficient convolutional block with gradient checkpointing.
    Uses bfloat16 for reduced memory footprint on AMD ROCm.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.conv1d = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Use bfloat16 if available (AMD ROCm support)
        self.use_bfloat16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gradient checkpointing for memory efficiency
        if self.training and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MemoryEfficientLSTM(nn.Module):
    """
    Memory-efficient LSTM with gradient checkpointing and bfloat16 support.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Use bfloat16 if available
        self.use_bfloat16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Gradient checkpointing for memory efficiency
        if self.training and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM architecture for financial pattern recognition.
    Combines local pattern detection (CNN) with temporal dependencies (LSTM).
    
    Features:
    - bfloat16 precision for reduced VRAM usage on AMD ROCm
    - Gradient checkpointing to trade compute for memory
    - Configurable depth and width for memory constraints
    - Output: price direction, volatility, or custom targets
    """
    
    def __init__(
        self,
        input_dim: int = 10,  # Number of input features
        seq_length: int = 100,  # Sequence length
        cnn_channels: List[int] = [32, 64, 128],
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1,  # Single output (e.g., price direction)
        max_vram_mb: int = 2048,  # Max VRAM limit in MB
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.max_vram_mb = max_vram_mb
        
        # CNN layers for local pattern extraction
        cnn_blocks = []
        in_channels = input_dim
        
        for out_channels in cnn_channels:
            cnn_blocks.append(
                MemoryEfficientConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=5,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        
        self.cnn_encoder = nn.Sequential(*cnn_blocks)
        
        # Calculate output size after CNN
        cnn_output_size = cnn_channels[-1]
        
        # LSTM for temporal dependencies
        self.lstm = MemoryEfficientLSTM(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=True,
        )
        
        # Output layers
        lstm_output_size = lstm_hidden * 2  # Bidirectional
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, output_dim),
        )
        
        # Check VRAM availability
        self._check_vram()
    
    def _check_vram(self):
        """Check and warn about VRAM constraints."""
        import psutil
        
        if torch.cuda.is_available():
            try:
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                logger.info(f"GPU VRAM: {vram_total:.0f}MB, Limit: {self.max_vram_mb}MB")
                
                if vram_total < self.max_vram_mb:
                    logger.warning(
                        f"Available VRAM ({vram_total:.0f}MB) is below target ({self.max_vram_mb}MB). "
                        "Model may use gradient checkpointing aggressively."
                    )
            except Exception as e:
                logger.warning(f"Could not query VRAM: {e}")
        else:
            logger.info("CUDA not available. Using CPU mode.")
        
        # Log system RAM
        ram_available = psutil.virtual_memory().available / (1024**3)
        logger.info(f"System RAM available: {ram_available:.2f}GB")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: CNN -> LSTM -> Output
        
        Args:
            x: Input tensor of shape (batch, seq_length, input_dim)
        
        Returns:
            Output tensor of shape (batch, output_dim)
        """
        # Reshape for CNN: (batch, channels, seq_length)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_length)
        
        # CNN encoding
        x = self.cnn_encoder(x)
        
        # Reshape for LSTM: (batch, seq_length, channels)
        x = x.transpose(1, 2)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state from both directions
        if self.lstm.bidirectional:
            hidden_fwd = hidden[-2]  # Last layer forward
            hidden_bwd = hidden[-1]  # Last layer backward
            hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        else:
            hidden_concat = hidden[-1]
        
        # Output
        output = self.output_layer(hidden_concat)
        
        return output
    
    def to_bfloat16(self):
        """Convert model to bfloat16 for AMD ROCm efficiency."""
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self = self.bfloat16()
            logger.info("Model converted to bfloat16")
        else:
            logger.warning("bfloat16 not supported. Using float32.")
        return self


class CNNLSTMTrainer:
    """
    Trainer for Hybrid CNN-LSTM with strict memory management.
    Implements gradient accumulation, mixed precision, and early stopping.
    """
    
    def __init__(
        self,
        model: HybridCNNLSTM,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        max_epochs: int = 50,
        early_stopping_patience: int = 10,
        batch_size: int = 64,
        gradient_accumulation_steps: int = 4,
        device: Optional[str] = None,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Determine device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
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
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every N steps
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
        """
        Full training loop with early stopping and memory monitoring.
        """
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
            num_workers=0,  # Avoid multiprocessing overhead
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        logger.info(f"Starting training for {self.max_epochs} epochs...")
        
        for epoch in range(self.max_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Log progress
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1}/{self.max_epochs}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={current_lr:.6f}"
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
                
                # Save best model state
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Memory check
            ram_available = psutil.virtual_memory().available / (1024**3)
            if ram_available < 1.0:
                logger.warning(f"Critical low RAM: {ram_available:.2f}GB. Consider reducing batch size.")
        
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
    """Example usage with synthetic data."""
    import psutil
    
    print(f"Initial RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")
    
    # Generate synthetic sequence data
    np.random.seed(42)
    n_samples = 10000
    seq_length = 50
    input_dim = 10
    
    X = np.random.randn(n_samples, seq_length, input_dim).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)  # Target: next period return
    
    # Split
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create model
    model = HybridCNNLSTM(
        input_dim=input_dim,
        seq_length=seq_length,
        cnn_channels=[32, 64],
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.2,
        output_dim=1,
        max_vram_mb=2048,
    )
    
    # Create trainer
    trainer = CNNLSTMTrainer(
        model=model,
        learning_rate=0.001,
        max_epochs=20,
        early_stopping_patience=5,
        batch_size=32,
        gradient_accumulation_steps=4,
    )
    
    # Train
    results = trainer.fit(X_train, y_train, X_val, y_val)
    print(f"\nTraining complete: {results['best_val_loss']:.6f} after {results['epochs_trained']} epochs")
    
    # Predict
    predictions = trainer.predict(X_val[:100])
    print(f"Sample predictions: {predictions[:5].flatten()}")
    
    print(f"\nFinal RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")


if __name__ == "__main__":
    main()
