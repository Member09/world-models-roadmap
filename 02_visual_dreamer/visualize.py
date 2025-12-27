import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(model, loader):
    """
    Plots original vs reconstructed images and the latent space.
    """
    model.eval() 
    
    # 1. Get one batch of data
    # Handle the tuple/list wrapping from TensorDataset
    batch = next(iter(loader))
    if isinstance(batch, (list, tuple)):
        data = batch[0]
    else:
        data = batch
        
    # Flatten dynamically based on model input
    input_dim = model.encoder.input_dim
    data = data.view(-1, input_dim)
    
    # 2. Reconstruct
    with torch.no_grad():
        recon, mu, _ = model(data)
    
    # Move to CPU for plotting
    data = data.cpu().numpy()
    recon = recon.cpu().numpy()
    
    # 3. Plot Reconstructions
    n_samples = 8
    plt.figure(figsize=(12, 4))
    
    # Calculate side size (e.g., sqrt(1024) = 32)
    side = int(np.sqrt(input_dim))
    
    for i in range(n_samples):
        # Original
        ax = plt.subplot(2, n_samples, i + 1)
        plt.imshow(data[i].reshape(side, side), cmap='gray')
        plt.axis('off')
        if i == 0: ax.set_title("Reality")
        
        # Reconstruction
        ax = plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(recon[i].reshape(side, side), cmap='gray')
        plt.axis('off')
        if i == 0: ax.set_title("Dream")
    
    plt.tight_layout()
    plt.show()

    print("Visualization complete.")