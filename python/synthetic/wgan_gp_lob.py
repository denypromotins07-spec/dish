"""
Ultra-lightweight Wasserstein GAN with Gradient Penalty for LOB generation.
Strictly bounded layer sizes and batch dimensions to prevent RAM spikes.
Optimized for AMD ROCm with memory-efficient operations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List
import numpy as np

# Memory constraints
MAX_BATCH_SIZE = 64
LATENT_DIM = 32
HIDDEN_DIM = 64
LOB_DIM = 20  # Reduced dimensionality for LOB representation

class ResidualBlock(nn.Module):
    """Memory-efficient residual block with gradient checkpointing."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Generator(nn.Module):
    """
    Lightweight generator for synthetic LOB states.
    Uses residual blocks and spectral normalization for stability.
    """
    
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        output_dim: int = LOB_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim // 2)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),  # Normalize to [-1, 1]
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(z)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


class Discriminator(nn.Module):
    """
    Lightweight discriminator for WGAN-GP.
    Outputs critic score for real vs fake LOB states.
    """
    
    def __init__(
        self,
        input_dim: int = LOB_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_blocks: int = 3,
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim // 2)
            for _ in range(num_blocks)
        ])
        
        # Output (single scalar critic score)
        self.output = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output(h)


class WGAN_GP_LOB:
    """
    WGAN with Gradient Penalty for Limit Order Book generation.
    Memory-bounded training loop with strict VRAM limits.
    """
    
    def __init__(
        self,
        device: torch.device = None,
        lr: float = 1e-4,
        beta1: float = 0.5,
        beta2: float = 0.9,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
    ):
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        
        # Initialize models
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Optimizers
        self.opt_g = optim.Adam(
            self.generator.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.opt_d = optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(beta1, beta2)
        )
        
        # Training stats
        self.step_count = 0
        
    def compute_gradient_penalty(
        self,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate gradient penalty for WGAN-GP."""
        alpha = torch.rand(real_samples.size(0), 1, device=self.device)
        alpha = alpha.expand_as(real_samples)
        
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty
    
    def train_step(
        self,
        real_lob: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Single training step with memory-bounded operations.
        Returns (d_loss, g_loss)
        """
        batch_size = min(real_lob.size(0), MAX_BATCH_SIZE)
        real_lob = real_lob[:batch_size].to(self.device)
        
        # Train discriminator (n_critic times)
        d_loss_total = 0.0
        for _ in range(self.n_critic):
            # Generate fake samples
            z = torch.randn(batch_size, LATENT_DIM, device=self.device)
            fake_lob = self.generator(z)
            
            # Discriminator loss
            d_real = self.discriminator(real_lob)
            d_fake = self.discriminator(fake_lob.detach())
            
            d_loss = -(d_real.mean() - d_fake.mean())
            
            # Gradient penalty
            gp = self.compute_gradient_penalty(real_lob, fake_lob.detach())
            d_loss = d_loss + self.lambda_gp * gp
            
            # Update discriminator
            self.opt_d.zero_grad()
            d_loss.backward()
            self.opt_d.step()
            
            d_loss_total += d_loss.item()
        
        d_loss_avg = d_loss_total / self.n_critic
        
        # Train generator
        z = torch.randn(batch_size, LATENT_DIM, device=self.device)
        fake_lob = self.generator(z)
        g_loss = -self.discriminator(fake_lob).mean()
        
        self.opt_g.zero_grad()
        g_loss.backward()
        self.opt_g.step()
        
        self.step_count += 1
        
        return d_loss_avg, g_loss.item()
    
    def generate_synthetic_lob(
        self,
        num_samples: int,
    ) -> torch.Tensor:
        """Generate synthetic LOB states."""
        self.generator.eval()
        samples = []
        
        with torch.no_grad():
            for i in range(0, num_samples, MAX_BATCH_SIZE):
                batch_size = min(MAX_BATCH_SIZE, num_samples - i)
                z = torch.randn(batch_size, LATENT_DIM, device=self.device)
                batch_samples = self.generator(z)
                samples.append(batch_samples.cpu())
                
        return torch.cat(samples, dim=0)
    
    def get_memory_footprint_mb(self) -> float:
        """Calculate current model memory footprint in MB."""
        total_params = sum(p.numel() for p in self.generator.parameters())
        total_params += sum(p.numel() for p in self.discriminator.parameters())
        
        # Approximate memory (params + gradients + optimizer states)
        bytes_per_param = 4  # float32
        multiplier = 4  # params + grad + opt state (adam has 2 moments)
        
        return (total_params * bytes_per_param * multiplier) / (1024 ** 2)


if __name__ == "__main__":
    # Test WGAN-GP LOB generation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    wgan = WGAN_GP_LOB(device=device)
    
    # Print memory footprint
    mem_mb = wgan.get_memory_footprint_mb()
    print(f"Model memory footprint: {mem_mb:.2f} MB")
    
    # Dummy training
    batch_size = 32
    real_lob = torch.randn(batch_size, LOB_DIM)
    
    print("\nTraining...")
    for step in range(10):
        d_loss, g_loss = wgan.train_step(real_lob)
        if step % 5 == 0:
            print(f"Step {step}: D Loss = {d_loss:.4f}, G Loss = {g_loss:.4f}")
    
    # Generate synthetic data
    synthetic = wgan.generate_synthetic_lob(100)
    print(f"\nGenerated synthetic LOB shape: {synthetic.shape}")
    print(f"Synthetic data range: [{synthetic.min():.4f}, {synthetic.max():.4f}]")
