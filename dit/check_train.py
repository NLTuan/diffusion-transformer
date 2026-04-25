import torch
from dit import DiT_tiny

def test_training_step():
    model = DiT_tiny()
    model.train() # Set to train mode
    
    batch_size = 4
    latent = torch.randn(batch_size, 4, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))
    y = torch.randint(0, 10, (batch_size,))
    
    out = model(latent, t, y)
    
    assert out.shape == latent.shape, f"Output shape {out.shape} does not match input shape {latent.shape}"
    
    target = torch.randn_like(out)
    loss = torch.nn.functional.mse_loss(out, target)
    
    try:
        loss.backward()
    except Exception as e:
        print(f"Backward pass failed: {e}")
        return False
        
    has_grads = any(p.grad is not None for p in model.parameters())
    if not has_grads:
        print("Backward pass completed but no gradients found!")
        return False
        
    print("Forward and backward pass completed successfully. Gradients are flowing.")
    return True

if __name__ == "__main__":
    success = test_training_step()
    if success:
        print("Model is ready for training!")
    else:
        print("Model check failed.")
