import torch
import cuda_tensor_loader

# Load a tensor from a .pth file to CUDA
tensor = cuda_tensor_loader.load_pth_tensor_to_cuda("path/to/your/tensor.pth", device_id=0)

print(f"Tensor loaded successfully.")
print(f"Shape: {tensor.shape}")
print(f"Device: {tensor.device}")

# Use the tensor in your PyTorch code...

# When you're done, you can move it back to CPU if needed
tensor = tensor.cpu()

# Clear CUDA cache
torch.cuda.empty_cache()
