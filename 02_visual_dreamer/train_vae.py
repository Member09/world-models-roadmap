import torch
import torch.nn as nn
import os

# Sibling imports 
from vae import VAE
from dataset import get_vae_data_loader
from visualize import visualize_results

# --- 1. The Loss Function ---
def elbo_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- 2. The Training Loop ---
def train(model, loader, epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    input_dim = model.encoder.input_dim 

    print(f"Starting VAE Training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, batch in enumerate(loader):
            # Handle the list/tuple wrapping from TensorDataset
            if isinstance(batch, (list, tuple)):
                data = batch[0]
            else:
                data = batch
            
            data = data.view(-1, input_dim)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = elbo_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print(f"Epoch {epoch}: Loss = {train_loss / len(loader.dataset):.4f}")

    return model

# --- 3. The Execution Block ---
if __name__ == "__main__":
    # A. Setup
    print("Initializing Physics VAE...")
    loader = get_vae_data_loader(batch_size=128)
    
    # B. Init Model (32x32 = 1024 pixels)
    model = VAE(input_dim=1024, latent_dim=2)
    
    # C. Train
    model = train(model, loader, epochs=20)

    # D. Visualize the results
    print("Visualizing performance...")
    visualize_results(model, loader)
    
    # E. Save Model
    save_path = os.path.join("..", "saved_models", "vae_ball.pth")
    # Ensure folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")