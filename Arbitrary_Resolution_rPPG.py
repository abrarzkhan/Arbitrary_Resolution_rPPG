import torch
import torch.nn.functional as F
import sys

print("1. Creating dummy tensors and moving to GPU...")
sys.stdout.flush()
input_tensor = torch.randn(4, 3, 16, 16).cuda()
grid = torch.clamp(torch.randn(4, 16, 16, 2), -1, 1).cuda()

print("2. Running grid_sample on RTX 5070 Ti...")
sys.stdout.flush()
# This is the exact math operation that is likely freezing your laptop
out = F.grid_sample(input_tensor, grid, align_corners=True)
torch.cuda.synchronize()

print("3. SUCCESS! The GPU can do the math.")
sys.stdout.flush()