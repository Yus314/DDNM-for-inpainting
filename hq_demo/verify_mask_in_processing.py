import torch
import numpy as np
from PIL import Image

# Simulate the mask processing in the test
mask_path = "data/datasets/gt_keep_masks/center_mask_512/center_mask_512x512.png"
pil_mask = Image.open(mask_path).convert('RGB')

# Test's inversion
arr_mask = np.array(pil_mask).astype(np.float32) / 255.0
arr_mask = 1.0 - arr_mask  # Inversion

print("=== After test inversion ===")
print(f"Center value: {arr_mask[256, 256, 0]:.3f} (should be 0.0 for inpainting)")
print(f"Edge value: {arr_mask[50, 50, 0]:.3f} (should be 1.0 for keeping)")

# Convert to tensor (as in test)
gt_keep_mask = torch.from_numpy(np.transpose(arr_mask, [2, 0, 1])).unsqueeze(0)
print(f"\nTensor shape: {gt_keep_mask.shape}")
print(f"Tensor center value: {gt_keep_mask[0, 0, 256, 256].item():.3f}")
print(f"Tensor edge value: {gt_keep_mask[0, 0, 50, 50].item():.3f}")

# What DDNM actually receives
print("\n=== DDNM receives ===")
print(f"mask (center): {gt_keep_mask[0, 0, 256, 256].item():.3f}")
print(f"mask (edge): {gt_keep_mask[0, 0, 50, 50].item():.3f}")
print("\nThis is CORRECT: 0.0 for inpaint, 1.0 for keep")

# But what gets saved?
print("\n=== What tensor2im does ===")
print(f"Input mask value: 0.0 (inpaint region)")
print(f"tensor2im: (0.0 + 1) / 2 = 0.5 → 127 in uint8")
print(f"Result: Mask becomes GRAY instead of BLACK")
print(f"\nInput mask value: 1.0 (keep region)")
print(f"tensor2im: (1.0 + 1) / 2 = 1.0 → 255 in uint8")
print(f"Result: Correct (white)")
