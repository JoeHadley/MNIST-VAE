
import torch
import torch.nn as nn


#import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset


from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid


import kagglehub

# Download latest version
path = kagglehub.dataset_download("hojjatk/mnist-dataset")

print("Path to dataset files:", path)


class VAE(nn.Module):
  def __init__(self, input_dim=28*28, hidden_dim=16, latent_dim=2, device='cpu',beta=1.0,lr=1e-3):
    super(VAE, self).__init__()
    self.device = device   
    self.beta = beta
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.latent_dim = latent_dim
    
    

    # Encoder
    self.encoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_dim, latent_dim),
        nn.LeakyReLU(0.2)
    )
    

    # Latent mean and variance
    self.mean_layer = nn.Linear(latent_dim, latent_dim)
    self.logvar_layer = nn.Linear(latent_dim, latent_dim)

    # Decoder
    self.decoder = nn.Sequential(
      nn.Linear(latent_dim, hidden_dim),
      nn.LeakyReLU(0.2),
      nn.Linear(hidden_dim, input_dim)
    )


    self.optimizer = optim.Adam(self.parameters(), lr=lr)

    self.to(self.device)


  def encode(self, x):
      
    h = self.encoder(x)
    mean = self.mean_layer(h)
    logvar = self.logvar_layer(h)
    return mean, logvar

  def reparameterization(self, mean, logvar):
      
    logvar = torch.clamp(logvar, min=-20, max=20)  # Clamp logvar for numerical stability
    std = torch.exp(0.5 * logvar)  # Calculate the standard deviation
    epsilon = torch.randn_like(std)  # Sample epsilon
    z = mean + std * epsilon  # Reparameterization trick
    return z

  def decode(self, z):
    return self.decoder(z)

  def forward(self, x):
    """
    Full pass: returns reconstruction, mean, logvar
    """
    mean, logvar = self.encode(x)
    z = self.reparameterization(mean, logvar)
    recon = self.decode(z)
    return recon, mean, logvar


  def computeKLLoss(self, mean, logvar):
    """
    Standard Gaussian KL divergence between q(z|x)=N(mean, var) and p(z)=N(0,I)
    Returns scalar for the batch (sum over latent dims, mean over batch)
    """
    # KL per data point: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
    # average over batch
    kl = torch.mean(kl_per_sample)
    return self.beta * kl

  def reconstruction_loss(self, recon, x, reduction='mean'):
    """
    MSE reconstruction loss. Use reduction='sum' or 'mean'
    recon and x shapes: (batch_size, input_dim)
    """
    # you can swap to BCE if input in [0,1] and you prefer that.
    mse = nn.functional.mse_loss(recon, x, reduction=reduction)
    return mse

  def compute_loss(self, x, recon, mean, logvar, recon_reduction='mean'):
    """
    Combined loss: recon + beta * KL
    returns: total_loss, recon_loss, kl_loss
    """
    recon_loss = self.reconstruction_loss(recon, x, reduction=recon_reduction)
    kl_loss = self.computeKLLoss(mean, logvar)
    # If recon_reduction == 'mean', both are comparable; else you might want to scale.
    total = recon_loss + kl_loss
    return total, recon_loss, kl_loss

  def log_prior(self, z, mu0=0.0, sigma0=1.0):
    """
    Log prior p(z) assuming isotropic Gaussian N(mu0, sigma0^2).
    Returns sum log p(z) over batch (scalar).
    """
    var0 = sigma0**2
    d = z.shape[1]
    log_const = -0.5 * d * torch.log(2 * torch.pi * var0)
    quad = -0.5 * torch.sum((z - mu0)**2, dim=1) / var0
    return torch.mean(log_const + quad)
    
  def log_likelihood(self,phi, data, sigma_y):
    """
    Gaussian likelihood: p(data | phi) = N(data | g(phi), sigma_y^2 I)
    - forward_model(phi): predicts mean of data given phi
    """
    
    pred = self.forward(phi)  # Forward pass to get predictions

    
    var_y = sigma_y**2
    diff = data - pred
    d = data.shape[0]
    log_term = torch.log(torch.tensor(2.0 * torch.pi * var_y, device=diff.device))
    logl = -0.5 * (d * log_term + (diff ** 2).sum() / var_y)
    return logl
  
  def step(self, x):
    """
    One optimization step on batch x (expects x already on correct device)
    returns loss tuple
    """
    self.optimizer.zero_grad()
    recon, mean, logvar = self.forward(x)
    total, recon_loss, kl_loss = self.compute_loss(x, recon, mean, logvar, recon_reduction='mean')
    total.backward()
    self.optimizer.step()
    return total.item(), recon_loss.item(), kl_loss.item()


# Example usage:
# - dataset: PyTorch dataset that yields (image, label). We flatten image to vector.
# - make sure image values are floats and normalized (e.g., [0,1]).


def save_vae(model, path="vae.pth"):
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "latent_dim": model.latent_dim,
        "beta": model.beta,
    }, path)
    print(f"VAE saved to {path}")


def load_vae(path, device="cpu"):
    print(f"Loading VAE from {path}...")
    checkpoint = torch.load(path, map_location=device)
    model = VAE(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
        device=device,
        beta=checkpoint["beta"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print("VAE loaded.")
    return model

