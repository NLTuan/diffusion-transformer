# diffusion-transformer
Custom reimplementation of a Diffusion Transformer (DiT), building a diffusion model entirely with transformer blocks.

Training has been configured for the MNIST and CIFAR-10 datasets.

## Environment Setup

This project uses `uv` for dependency management.

1. Ensure you have `uv` installed on your system.
2. Resolve and install the dependencies to a virtual environment:
   ```bash
   uv sync
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Running the Training

To train the model on a dataset, run the appropriate training script from the root of the repository.

- **MNIST dataset**:
  ```bash
  python training/train_mnist.py
  ```

- **CIFAR-10 dataset**:
  ```bash
  python training/train_cifar10.py
  ```

Model generation sample scripts are also available, ensuring the environment is activated.