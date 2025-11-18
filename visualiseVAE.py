import matplotlib.pyplot as plt
import numpy as np

import torch

from VAEDefinition import VAE
from VAEDefinition import load_vae

def plot_latent_space(model, grid_size=20, z_range=3.0):
    """
    Visualizes decoder behavior over the 2D latent space.
    Produces a grid_size x grid_size image of decoded samples.
    """
    assert model.latent_dim == 2, "Latent dimension must be 2 for visualization!"

    print("Checkpoint1")
    model.eval()
    device = model.device

    # Create a grid of latent coordinates
    lin = np.linspace(-z_range, z_range, grid_size)
    z_points = np.array([[x, y] for y in lin for x in lin], dtype=np.float32)
    z_tensor = torch.tensor(z_points, device=device)

    # Decode
    with torch.no_grad():
        decoded = model.decode(z_tensor).cpu()

    # Reshape into image grid
    img_dim = int(np.sqrt(model.input_dim))  # assumes square images
    decoded = decoded.view(grid_size * grid_size, img_dim, img_dim)

    # Build one big canvas
    canvas = np.zeros((grid_size * img_dim, grid_size * img_dim))
    idx = 0

    for i in range(grid_size):
        for j in range(grid_size):
            canvas[i*img_dim:(i+1)*img_dim, j*img_dim:(j+1)*img_dim] = decoded[idx]
            idx += 1

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas, cmap="gray")
    plt.title("VAE latent space manifold")
    plt.axis("off")
    plt.show()





if __name__ == "__main__":
    vae_model = load_vae("vae.pth", device='cpu')
    plot_latent_space(vae_model, grid_size=20, z_range=3.0)

