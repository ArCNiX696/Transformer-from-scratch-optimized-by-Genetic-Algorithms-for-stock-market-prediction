#!/usr/bin/env python3

import torch

def check_cuda_and_run_tensor_operation():
    """
    Checks if CUDA is available via PyTorch and performs a simple
    tensor operation on the GPU (if available).
    """
    if torch.cuda.is_available():
        print("CUDA is installed and active!")
        # Create two tensors on the GPU
        a = torch.randn((3, 3), device="cuda")
        b = torch.randn((3, 3), device="cuda")
        # Perform a simple operation
        c = a @ b
        print("Tensor operation on GPU (matrix multiplication) succeeded.")
        print("Result:\n", c)
    else:
        print("CUDA is NOT available on this machine.")

if __name__ == "__main__":
    check_cuda_and_run_tensor_operation()
