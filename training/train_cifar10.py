import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from dit.dit import DiT_tiny
import os
from tqdm import tqdm
from dataclasses import dataclass
from torchvision.utils import make_grid
from PIL import Image

@dataclass
class TrainingConfig:
    image_size: int = 32
    in_channels: int = 3
    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    num_timesteps: int = 1000
    mixed_precision: str = "fp16" # 'no', 'fp16', 'bf16'
    num_workers: int = 4
    output_dir: str = "./outputs/cifar10_dit_checkpoints"
    save_model_epochs: int = 5
    objective: str = "flow_matching" # 'ddpm' or 'flow_matching'
    hf_repo_id: str = "NLTuan/cifar10_dit_flow" # e.g., 'username/my-cifar10-dit'

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
        """DDIM step."""
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0, device=self.device)
        pred_original_sample = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev) * model_output
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        return prev_sample

@torch.no_grad()
def evaluate_and_save_video(model, config, epoch, device):
    model.eval()
    batch_size = 16
    labels = torch.randint(0, 10, (batch_size,), device=device)
    image = torch.randn((batch_size, config.in_channels, config.image_size, config.image_size), device=device)
    frames = []
    
    if config.objective == "ddpm":
        scheduler = DDIMScheduler(num_train_timesteps=config.num_timesteps, device=device)
        num_inference_steps = 50
        scheduler.set_timesteps(num_inference_steps)
        for t in scheduler.timesteps:
            timesteps_vec = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = model(image, timesteps_vec, y=labels)
            image = scheduler.step(noise_pred, t, image)
            grid = make_grid((image / 2 + 0.5).clamp(0, 1), nrow=4)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            frames.append(Image.fromarray(ndarr))
    elif config.objective == "flow_matching":
        num_inference_steps = 50
        t_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
        for i in range(num_inference_steps):
            t_current = t_steps[i]
            t_next = t_steps[i+1]
            dt = t_next - t_current
            timesteps_vec = (torch.full((batch_size,), t_current, device=device) * 1000).long()
            v_pred = model(image, timesteps_vec, y=labels)
            image = image + v_pred * dt
            grid = make_grid((image / 2 + 0.5).clamp(0, 1), nrow=4)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            frames.append(Image.fromarray(ndarr))
            
    if frames:
        gif_path = os.path.join(config.output_dir, f"denoising_epoch_{epoch}.gif")
        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0
        )
        print(f"Saved evaluation video to {gif_path}")
        model.train()
        return gif_path
    model.train()
    return None

class CIFAR10Dataset(torch.utils.data.Dataset):
    """Wrapper to properly transform HF Dataset."""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["img"]
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
    hf_dataset = load_dataset("cifar10", split="train")
    
    # CIFAR-10 is already 32x32, no padding needed.
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), # Basic augmentation useful for CIFAR
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # map ranges from [0, 1] to [-1, 1] for 3 channels
    ])
    
    dataset = CIFAR10Dataset(hf_dataset, transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    
    # 2. Model
    # We use in_channels=3 for RGB
    model = DiT_tiny(input_size=config.image_size, in_channels=config.in_channels).to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 3. Optimizer & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    use_amp = config.mixed_precision != "no" and device == "cuda"
    dtype = torch.float16 if config.mixed_precision == "fp16" else torch.bfloat16
    
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
            bsz = clean_images.shape[0]

            if config.objective == "ddpm":
                noise = torch.randn_like(clean_images)
                
                # Sample random timesteps
                timesteps = torch.randint(0, noise_scheduler.num_timesteps, (bsz,), device=device).long()
                
                # Add noise to clean images
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                
                # Target for DDPM/DDIM is the added noise (epsilon prediction)
                target = noise
            elif config.objective == "flow_matching":
                # 1. Sample target noise
                noise = torch.randn_like(clean_images)

                # 2. Sample continuous timesteps (t as a float between 0.0 and 1.0)
                t_continuous = torch.rand((bsz,), device=device)
                
                # We scale t up by 1000 so the DiT timestep embeddings can process it properly
                timesteps = (t_continuous * 1000).long()

                # Reshape t for broadcasting -> shape becomes (bsz, 1, 1, 1)
                t_expanded = t_continuous.view(bsz, 1, 1, 1)

                # 3. Create linear interpolation for the noisy images
                noisy_images = (1.0 - t_expanded) * clean_images + t_expanded * noise

                # 4. Target is the velocity pointing from clean_image to noise
                target = noise - clean_images
            else:
                raise ValueError(f"Unknown training objective: {config.objective}")
            
            optimizer.zero_grad()
            
            # Predict
            with torch.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=dtype, enabled=use_amp):
                # Our DiT outputs either predicted noise (DDPM) or vector field (Flow Matching)
                pred = model(noisy_images, timesteps, y=labels)
                loss = loss_fn(pred, target)
                
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
            model_path = os.path.join(config.output_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path}")
            
            # Evaluate and save video
            gif_path = evaluate_and_save_video(model, config, epoch+1, device)
            
            if config.hf_repo_id and gif_path:
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()
                    
                    # Ensure the repository exists before pushing
                    api.create_repo(repo_id=config.hf_repo_id, exist_ok=True)
                    
                    api.upload_file(
                        path_or_fileobj=model_path,
                        path_in_repo=f"model_epoch_{epoch+1}.pt",
                        repo_id=config.hf_repo_id,
                        repo_type="model"
                    )
                    api.upload_file(
                        path_or_fileobj=gif_path,
                        path_in_repo=f"denoising_epoch_{epoch+1}.gif",
                        repo_id=config.hf_repo_id,
                        repo_type="model"
                    )
                    print(f"Pushed model and video to Hugging Face Hub ({config.hf_repo_id})")
                except Exception as e:
                    print(f"Failed to push to Hugging Face Hub: {e}")

if __name__ == "__main__":
    main()
