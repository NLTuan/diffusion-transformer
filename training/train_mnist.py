import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from dit.dit import DiT_tiny
import os
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size: int = 32
    in_channels: int = 1
    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    num_timesteps: int = 1000
    mixed_precision: str = "fp16" # 'no', 'fp16', 'bf16'
    num_workers: int = 4
    output_dir: str = "mnist_dit_checkpoints"
    save_model_epochs: int = 5

class DDPMScheduler:
    """A minimal DDPMScheduler for the forward diffusion process."""
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, original_samples, noise, timesteps):
        """Adds noise to the original samples at specified timesteps."""
        b = original_samples.shape[0]
        alpha_cumprod_t = self.alphas_cumprod[timesteps].view(b, 1, 1, 1)
        
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod_t)
        
        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise
        return noisy_samples

class MNISTDataset(torch.utils.data.Dataset):
    """Wrapper to properly transform HF Dataset."""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def main():
    config = TrainingConfig()
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Dataset & Dataloader
    hf_dataset = load_dataset("mnist", split="train")
    
    # DiT patches images, so image_size should be divisible by patch_size. We pad 28x28 to 32x32.
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # map ranges from [0, 1] to [-1, 1]
    ])
    
    dataset = MNISTDataset(hf_dataset, transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    
    # 2. Model
    # We use in_channels=1 instead of default 4 because we operate on pixel space, not latents.
    model = DiT_tiny(input_size=config.image_size, in_channels=config.in_channels).to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 3. Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    use_amp = config.mixed_precision != "no" and device == "cuda"
    dtype = torch.float16 if config.mixed_precision == "fp16" else torch.bfloat16
    
    # GradScaler is no longer necessary/used for bfloat16, but helpful for float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and dtype == torch.float16))
    
    # 4. Noise Scheduler
    noise_scheduler = DDPMScheduler(num_timesteps=config.num_timesteps, device=device)
    
    loss_fn = nn.MSELoss()
    
    # 5. Training Loop
    print(f"Starting training for {config.num_epochs} epochs!")
    global_step = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        epoch_loss = 0.0
        
        for batch in progress_bar:
            clean_images, labels = batch
            clean_images = clean_images.to(device)
            labels = labels.to(device)
            
            # Sample noise
            noise = torch.randn_like(clean_images)
            
            # Sample random timesteps
            bsz = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (bsz,), device=device).long()
            
            # Add noise to clean images
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            optimizer.zero_grad()
            
            # Predict
            with torch.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=dtype, enabled=use_amp):
                # Our DiT outputs the predicted noise
                noise_pred = model(noisy_images, timesteps, y=labels)
                loss = loss_fn(noise_pred, noise)
                
            # Backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            global_step += 1
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % config.save_model_epochs == 0:
            torch.save(model.state_dict(), os.path.join(config.output_dir, f"model_epoch_{epoch+1}.pt"))
            print(f"Saved model to {config.output_dir}/model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()