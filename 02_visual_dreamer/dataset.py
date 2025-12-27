import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def generate_bouncing_ball_data(num_videos=1000, seq_len=30, size=30, r=2):
    """Generates a dataset of bouncing balls. 
       Output Shape: (num_videos, seq_len, 1, size, size)

    Args:
        num_videos (int, optional): _description_. Defaults to 1000.
        seq_len (int, optional): _description_. Defaults to 30.
        size (int, optional): _description_. Defaults to 30.
        r (int, optional): _description_. Defaults to 2.
    """
    print(f"Generating {num_videos} videos of bouncing balls...")

    # Initialize the Container (N,T,C,H,W)
    data = np.zeros((num_videos, seq_len, 1, size, size), dtype=np.float32)

    for i in range(num_videos):
        # 1. Random Start position with margins
        x = np.random.randint(r, size-r)
        y = np.random.randint(r, size-r)

        # 2, Random Velocity (-2 to +2 pixel per frame)
        vx = np.random.choice([-2,-1,1,2])
        vy = np.random.choice([-2,-1,1,2])

        for t in range(seq_len):
            # Draw the frame
            # Create grid of coordinates
            Y, X = np.ogrid[:size, :size]

            # calculate distance from center (x, y)
            dist = np.sqrt((X-x)**2 + (Y-y)**2)

            # Draw the ball (1.0 inside radius, 0.0 outside)
            frame = (dist <= r).astype(np.float32)
            data[i,t,0] = frame

            # Physics update
            x += vx
            y += vy

            # Collision Detection (Bounce)
            # If hitting left/right wall -> invert vx
            if x <= r or x >= size-r :
                vx = -vx
                x += vx * 2 # Un-stick from wall

            # If hitting Top/Bottom wall -> invert vy
            if y <= r or y >= size-r:
                vy = -vy
                y += vy * 2 # Un-stick from wall

    return torch.tensor(data)

def get_vae_data_loader(batch_size=64):
    """
    Prepares data specifically for VAE training.
    The VAE only cares about Single Frames, not Sequences.
    We flatten the videos into a pile of images.
    """
    # 1. Generate 500 videos of 30 frames
    videos = generate_bouncing_ball_data(num_videos=500, seq_len=30, size=32)
    
    # 2. Reshape for VAE: (500 * 30, 32*32)
    # We ignore the 'Sequence' and 'Channel' dims for now
    # New Shape: (15000, 1024)
    flattened_frames = videos.view(-1, 32*32)
    
    # 3. Create Loader
    dataset = TensorDataset(flattened_frames)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader 
