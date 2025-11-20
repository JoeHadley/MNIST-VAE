import torch
import torch.optim as optim
import torch.nn as nn
from VAEDefinition import VAE, save_vae
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VAE(input_dim=28*28, hidden_dim=256, latent_dim=10, device=device, beta=5.0, lr=1e-3)

# Prepare MNIST dataloader (flattened)
transform = transforms.Compose([
    transforms.ToTensor(),  # (C,H,W) in [0,1]
    transforms.Lambda(lambda t: t.view(-1))  # flatten to (28*28,)
])
train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

n_epochs = 5
for epoch in range(1, n_epochs+1):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)
        total, recon_loss, kl_loss = model.step(imgs)
        epoch_loss += total
        if batch_idx % 200 == 0:
            print(f"Epoch {epoch} batch {batch_idx}: total={total:.4f} recon={recon_loss:.4f} kl={kl_loss:.4f}")
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch} average loss: {avg_loss:.4f}")

save_vae(model, path="vae.pth")