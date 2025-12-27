# Building a World Model: From Scratch

This repository documents my journey building a Generative World Model (inspired by Ha & Schmidhuber, 2018). The goal is to build an agent that learns a compressed representation of its visual environment (VAE) and predicts future states (RNN) to "dream" potential outcomes.

## Roadmap

### Phase 1: Temporal Memory (RNN)
- **Goal:** Understand temporal dependencies and sequence prediction.
- **Task:** Trained an RNN to predict Sine Wave trajectories.
- **Outcome:** Model successfully learned frequency/amplitude patterns from scratch.
<!-- - [Link to Code](./01_memory_basics) -->

### Phase 2: Visual Compression (VAE)
- **Goal:** Learn a disentangled, compressed latent representation of visual data.
- **Task:** Implemented a Variational Autoencoder (VAE) with Reparameterization Trick.
- **Experiments:** - MNIST Digits (Static Reconstruction)
  - Synthetic Physics (Bouncing Balls)
- **Outcome:** Achieved stable reconstruction of physics objects in 32x32 grid.
<!-- - [Link to Code](./02_visual_dreamer) -->

### Phase 3: Latent Dynamics (In Progress)
- **Goal:** Connect VAE and RNN to simulate future video frames in latent space.


## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Generate Physics Data: `python 02_visual_dreamer/dataset.py`
3. Train VAE: `python 02_visual_dreamer/train_vae.py`