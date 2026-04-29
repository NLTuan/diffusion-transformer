import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from dit.dit import DiT_tiny
from training.train_mnist import TrainingConfig
import os

class DDIMScheduler:
    """A minimal DDIMScheduler for inference."""
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.num_train_timesteps = num_train_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def set_timesteps(self, num_inference_steps):
        """Sets the discrete timesteps used for the inference loop."""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = torch.flip((torch.arange(0, num_inference_steps) * step_ratio).round(), dims=[0]).long()
        self.timesteps = timesteps.to(self.device)
        
    def step(self, model_output, timestep, sample):
        """
        DDIM step.
        model_output: The predicted noise (epsilon) from the model.
        timestep: The current current timestep in the schedule.
        sample: The current noisy image x_t.
        """
        # 1. Get previous timestep
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        
        # 2. Get alphas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0, device=self.device)
        
        # 3. Compute predicted original sample (x_0) from x_t and predicted noise
        pred_original_sample = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        
        # 4. Compute next sample x_{t-1} using the deterministic DDIM equation
        # x_{t-1} = sqrt(alpha_prod_t_prev) * x_0 + sqrt(1 - alpha_prod_t_prev) * predicted_noise
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev) * model_output
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        
        return prev_sample

@torch.no_grad()
def main():
    config = TrainingConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Model (Make sure you point to a checkpoint you actually trained!)
    model = DiT_tiny(input_size=config.image_size, in_channels=config.in_channels).to(device)
    
    checkpoint_path = os.path.join(config.output_dir, "model_epoch_20.pt") # Change if needed
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained weights.")
        
    model.eval()
    
    # 2. Initialize DDIM Scheduler
    scheduler = DDIMScheduler(num_train_timesteps=config.num_timesteps, device=device)
    num_inference_steps = 50 # DDIM allows us to use much fewer steps!
    scheduler.set_timesteps(num_inference_steps)
    print(f"Running inference with {num_inference_steps} DDIM steps...")
    
    # 3. Sample
    batch_size = 16
    
    # Start with pure noise (x_T)
    image = torch.randn((batch_size, config.in_channels, config.image_size, config.image_size), device=device)
    
    # Generate conditional labels for MNIST (digits 0-9)
    # We'll just generate the first 16 digits: [0, 1, ..., 9, 0, 1, ..., 5]
    labels = torch.arange(batch_size, device=device) % 10
    print(f"Generating digits: {labels.cpu().tolist()}")
    
    # Sampling loop
    for t in scheduler.timesteps:
        # 1. Expand timestep vector
        timesteps_vec = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # 2. Predict noise
        # Note: If your DiT doesn't use classifier-free guidance, we just pass labels directly
        noise_pred = model(image, timesteps_vec, y=labels)
        
        # 3. Compute previous noisy sample x_{t-1}
        image = scheduler.step(noise_pred, t, image)
        
    # 4. Post-process and save image
    # The output image is in [-1, 1], so we map it back to [0, 1]
    image = (image / 2 + 0.5).clamp(0, 1)
    
    grid = make_grid(image, nrow=4)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(grid)
    plt.axis("off")
    plt.title(f"DDIM Generated Images ({num_inference_steps} steps)")
    plt.savefig("ddim_samples.png", bbox_inches='tight')
    print("Saved samples to ddim_samples.png!")

if __name__ == "__main__":
    main()
