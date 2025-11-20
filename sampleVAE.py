import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from VAEDefinition import VAE, load_vae


vae = load_vae("vae.pth", device='cpu')

latent_dim = vae.latent_dim

print(latent_dim)





z = torch.randn((1, latent_dim))

#decode and show the image
with torch.no_grad():
    decoded = vae.decode(z).cpu().view(28, 28).numpy()
plt.imshow(decoded, cmap='gray')
plt.title("Decoded image from random latent vector")

plt.show()