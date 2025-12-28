import numpy as np
from PIL import Image

# Check saved mask from patch
mask_path = "results/center_hole_inpainting_results/patches/patch_04/mask.png"
mask_img = Image.open(mask_path)
mask_arr = np.array(mask_img)

print("=== Patch 04 Mask Analysis ===")
print(f"Image mode: {mask_img.mode}")
print(f"Array shape: {mask_arr.shape}")
print(f"Array dtype: {mask_arr.dtype}")
print(f"Min value: {mask_arr.min()}")
print(f"Max value: {mask_arr.max()}")
print(f"Unique values: {np.unique(mask_arr)[:10]}")  # First 10 unique values
print(f"Mean value: {mask_arr.mean():.3f}")

# Check center region (should be inpaint region)
h, w = mask_arr.shape[:2]
center_mask = mask_arr[h//4:3*h//4, w//4:3*w//4]
print(f"\n=== Center Region (Inpaint Area) ===")
print(f"Center min: {center_mask.min()}")
print(f"Center max: {center_mask.max()}")
print(f"Center mean: {center_mask.mean():.3f}")
print(f"Center unique values: {np.unique(center_mask)}")

# Check edge region (should be keep region)
edge_mask = mask_arr[:h//4, :w//4]
print(f"\n=== Edge Region (Keep Area) ===")
print(f"Edge min: {edge_mask.min()}")
print(f"Edge max: {edge_mask.max()}")
print(f"Edge mean: {edge_mask.mean():.3f}")
print(f"Edge unique values: {np.unique(edge_mask)}")

# Check original input mask
print("\n=== Original Input Mask ===")
orig_mask_path = "data/datasets/gt_keep_masks/center_mask_512/center_mask_512x512.png"
orig_mask = Image.open(orig_mask_path)
orig_arr = np.array(orig_mask)
print(f"Original mask shape: {orig_arr.shape}")
print(f"Original mask dtype: {orig_arr.dtype}")
print(f"Original mask min: {orig_arr.min()}")
print(f"Original mask max: {orig_arr.max()}")
print(f"Original mask unique values (first 10): {np.unique(orig_arr)[:10]}")

# Check inverted mask used in test
print("\n=== After Inversion (in test) ===")
print("Test performs: arr_mask = 1.0 - arr_mask")
print(f"So if original center was {orig_arr[256, 256]} (white)")
print(f"After inversion: center becomes {1.0 - orig_arr[256, 256]/255.0:.3f}")
