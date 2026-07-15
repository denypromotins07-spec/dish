"""
TimeGAN implementation for high-frequency time-series sequence generation.
Generates realistic trades, quotes, and cancellations with capped replay buffer.
Memory-efficient design for strict RAM constraints.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, List
from collections import deque
import numpy as np

# Memory constraints
MAX_SEQUENCE_LENGTH = 50
LATENT_DIM = 24
HIDDEN_DIM = 48
FEATURE_DIM = 8  # Reduced feature set: price, vol, spread, etc.
MAX_BUFFER_SIZE = 1000  # Strict cap on replay buffer


class TimeSeriesEmbedder(nn.Module):
    """Embeds raw time-series features into latent space."""
    
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, feature_dim)
        batch_size, seq_len, _ = x.shape
        x = x.view(-1, x.shape[-1])
        embedded = self.embedding(x)
        return embedded.view(batch_size, seq_len, -1)


class GeneratorRNN(nn.Module):
    """RNN-based generator for time-series sequences."""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.rnn = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
        
    def forward(
        self,
        z: torch.Tensor,
        init_hidden: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate sequence from latent vectors.
        z shape: (batch, seq_len, latent_dim)
        """
        outputs, hidden = self.rnn(z, init_hidden)
        return self.output_layer(outputs), hidden


class DiscriminatorRNN(nn.Module):
    """RNN-based discriminator for real vs fake sequences."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.rnn(x)
        return self.output(h_n[-1])


class ConditionalGenerator(nn.Module):
    """Generator that conditions on historical context."""
    
    def __init__(
        self,
        context_dim: int,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.context_encoder = nn.Linear(context_dim, hidden_dim)
        self.generator = GeneratorRNN(
            latent_dim + hidden_dim, hidden_dim, output_dim
        )
        
    def forward(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate sequence conditioned on context.
        z shape: (batch, seq_len, latent_dim)
        context shape: (batch, context_dim)
        """
        ctx_encoded = self.context_encoder(context).unsqueeze(1)
        ctx_expanded = ctx_encoded.expand(-1, z.shape[1], -1)
        
        combined = torch.cat([z, ctx_expanded], dim=-1)
        output, _ = self.generator(combined)
        return output


class ReplayBuffer:
    """
    Memory-capped replay buffer for training sequences.
    Uses deque for automatic eviction of old samples.
    """
    
    def __init__(self, max_size: int = MAX_BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        
    def add(self, sequence: torch.Tensor) -> None:
        """Add sequence to buffer."""
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.detach().cpu()
        self.buffer.append(sequence)
        
    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample batch of sequences."""
        if len(self.buffer) < batch_size:
            indices = np.random.choice(
                len(self.buffer), batch_size, replace=True
            )
        else:
            indices = np.random.choice(
                len(self.buffer), batch_size, replace=False
            )
        
        samples = [self.buffer[i] for i in indices]
        return torch.stack(samples)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        if len(self.buffer) == 0:
            return 0.0
        
        sample = self.buffer[0]
        bytes_per_sample = sample.numel() * sample.element_size()
        total_bytes = bytes_per_sample * len(self.buffer)
        return total_bytes / (1024 ** 2)


class TimeGAN:
    """
    TimeGAN for high-frequency sequence generation.
    Includes embedding network, generator, discriminator, and supervisor.
    """
    
    def __init__(
        self,
        device: torch.device = None,
        lr: float = 1e-3,
        gamma: float = 0.95,  # Loss weighting
    ):
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.gamma = gamma
        
        # Networks
        self.embedder = TimeSeriesEmbedder(FEATURE_DIM, HIDDEN_DIM).to(self.device)
        self.generator = ConditionalGenerator(
            HIDDEN_DIM, LATENT_DIM, HIDDEN_DIM, FEATURE_DIM
        ).to(self.device)
        self.discriminator = DiscriminatorRNN(FEATURE_DIM, HIDDEN_DIM).to(self.device)
        self.supervisor = GeneratorRNN(HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM).to(self.device)
        
        # Optimizers
        self.opt_e = optim.Adam(self.embedder.parameters(), lr=lr)
        self.opt_g = optim.Adam(self.generator.parameters(), lr=lr)
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.opt_s = optim.Adam(self.supervisor.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(MAX_BUFFER_SIZE)
        
    def train_embedder(self, sequences: torch.Tensor) -> float:
        """Train embedder to reconstruct sequences."""
        sequences = sequences.to(self.device)
        
        self.opt_e.zero_grad()
        
        # Encode and decode
        embedded = self.embedder(sequences)
        reconstructed = self.supervisor(embedded)
        
        # Reconstruction loss
        recon_loss = nn.MSELoss()(reconstructed, embedded.detach())
        
        recon_loss.backward()
        self.opt_e.step()
        
        return recon_loss.item()
    
    def train_supervisor(self, sequences: torch.Tensor) -> float:
        """Train supervisor to predict next step in embedded space."""
        sequences = sequences.to(self.device)
        
        self.opt_s.zero_grad()
        
        embedded = self.embedder(sequences)
        predicted = self.supervisor(embedded[:, :-1, :])
        
        # Prediction loss
        pred_loss = nn.MSELoss()(predicted, embedded[:, 1:, :])
        
        pred_loss.backward()
        self.opt_s.step()
        
        return pred_loss.item()
    
    def train_discriminator(
        self,
        real_sequences: torch.Tensor,
    ) -> float:
        """Train discriminator to distinguish real from fake."""
        real_sequences = real_sequences.to(self.device)
        batch_size = real_sequences.shape[0]
        
        # Generate fake
        z = torch.randn(
            batch_size, real_sequences.shape[1], LATENT_DIM, device=self.device
        )
        context = self.embedder(real_sequences)[:, 0, :]  # First step as context
        
        with torch.no_grad():
            fake_sequences = self.generator(z, context)
        
        self.opt_d.zero_grad()
        
        # Discriminator losses
        d_real = self.discriminator(real_sequences)
        d_fake = self.discriminator(fake_sequences.detach())
        
        d_loss = -(
            torch.log(d_real + 1e-8).mean() +
            torch.log(1 - d_fake + 1e-8).mean()
        )
        
        d_loss.backward()
        self.opt_d.step()
        
        return d_loss.item()
    
    def train_generator(self, real_sequences: torch.Tensor) -> float:
        """Train generator to fool discriminator."""
        real_sequences = real_sequences.to(self.device)
        batch_size = real_sequences.shape[0]
        
        # Generate fake
        z = torch.randn(
            batch_size, real_sequences.shape[1], LATENT_DIM, device=self.device
        )
        context = self.embedder(real_sequences)[:, 0, :]
        
        fake_sequences = self.generator(z, context)
        
        self.opt_g.zero_grad()
        
        # Adversarial loss
        d_fake = self.discriminator(fake_sequences)
        adv_loss = -torch.log(d_fake + 1e-8).mean()
        
        # Embedding consistency loss
        embedded_fake = self.embedder(fake_sequences)
        embedded_real = self.embedder(real_sequences)
        emb_loss = nn.MSELoss()(embedded_fake, embedded_real)
        
        g_loss = adv_loss + self.gamma * emb_loss
        
        g_loss.backward()
        self.opt_g.step()
        
        return g_loss.item()
    
    def train_step(self, sequences: torch.Tensor) -> Dict[str, float]:
        """Full training step."""
        # Ensure sequences fit memory constraints
        if sequences.shape[1] > MAX_SEQUENCE_LENGTH:
            sequences = sequences[:, :MAX_SEQUENCE_LENGTH, :]
        
        # Add to replay buffer
        self.replay_buffer.add(sequences)
        
        losses = {}
        losses['embedder'] = self.train_embedder(sequences)
        losses['supervisor'] = self.train_supervisor(sequences)
        losses['discriminator'] = self.train_discriminator(sequences)
        losses['generator'] = self.train_generator(sequences)
        
        return losses
    
    def generate_sequence(
        self,
        context: torch.Tensor,
        seq_length: int = MAX_SEQUENCE_LENGTH,
    ) -> torch.Tensor:
        """Generate new sequence given context."""
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(1, seq_length, LATENT_DIM, device=self.device)
            generated = self.generator(z, context.unsqueeze(0))
            
        return generated.cpu().squeeze(0)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        return {
            'replay_buffer_mb': self.replay_buffer.memory_usage_mb(),
            'buffer_size': len(self.replay_buffer),
        }


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    timegan = TimeGAN(device=device)
    
    # Dummy sequences (batch, seq_len, features)
    batch_size = 16
    seq_len = 30
    sequences = torch.randn(batch_size, seq_len, FEATURE_DIM)
    
    print("\nTraining TimeGAN...")
    for step in range(10):
        losses = timegan.train_step(sequences)
        if step % 5 == 0:
            print(f"Step {step}:")
            for k, v in losses.items():
                print(f"  {k}: {v:.4f}")
    
    # Memory stats
    stats = timegan.get_memory_stats()
    print(f"\nMemory stats: {stats}")
    
    # Generate new sequence
    context = torch.randn(FEATURE_DIM)
    generated = timegan.generate_sequence(context, seq_length=20)
    print(f"\nGenerated sequence shape: {generated.shape}")
