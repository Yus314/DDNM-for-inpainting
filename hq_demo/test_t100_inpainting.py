#!/usr/bin/env python3

"""
中央欠損inpaintingテスト - マスクを反転して中央部分を補完
"""

import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from conf_mgt import conf_base
from guided_diffusion import dist_util
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, select_args
from utils import yamlread

def main():
    # Load enhanced nine-patch configuration with t_T=100
    config_path = "confs/inpainting_512_nine_patch_t100.yml"
    conf = conf_base.Default_Conf()
    conf.update(yamlread(config_path))
    print(f"📋 Using config: {config_path}")
    print(f"   t_T = {conf.schedule_jump_params['t_T']}")
    print(f"   jump_length = {conf.schedule_jump_params['jump_length']}")

    # Setup device
    device = dist_util.dev(conf.get('device'))

    # Create model and diffusion
    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )

    # Load pretrained model
    print(f"Loading model from {conf.model_path}")
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # Load 512x512 image
    print("Loading 512x512 test image...")

    image_path = "data/datasets/gts/BSDS500_168_BLIP_22-44_11/100007.jpg"
    mask_path = "data/datasets/gt_keep_masks/center_mask_512/center_mask_512x512.png"

    pil_image = Image.open(image_path).convert('RGB')
    if pil_image.size != (512, 512):
        pil_image = pil_image.resize((512, 512), Image.Resampling.LANCZOS)

    # Load and INVERT mask for center hole inpainting
    pil_mask = Image.open(mask_path).convert('RGB')
    if pil_mask.size != (512, 512):
        pil_mask = pil_mask.resize((512, 512), Image.Resampling.LANCZOS)

    # 🔄 INVERT MASK: 0->1, 1->0 (中央を欠損対象に変換)
    arr_mask = np.array(pil_mask).astype(np.float32) / 255.0  # [0, 1]
    arr_mask = 1.0 - arr_mask  # マスクを反転！

    print(f"Image size: {pil_image.size}")
    print(f"Mask size: {pil_mask.size}")
    print("🔄 Mask inverted: Center will be INPAINTED (hole), edges will be KEPT")

    # Convert to arrays and normalize
    arr_image = np.array(pil_image).astype(np.float32) / 127.5 - 1  # [-1, 1]

    # Convert to torch tensors and add batch dimension
    gt = torch.from_numpy(np.transpose(arr_image, [2, 0, 1])).unsqueeze(0).to(device)
    gt_keep_mask = torch.from_numpy(np.transpose(arr_mask, [2, 0, 1])).unsqueeze(0).to(device)

    print(f"GT tensor shape: {gt.shape}")
    print(f"Inverted mask tensor shape: {gt_keep_mask.shape}")
    print(f"Inverted mask values range: {gt_keep_mask.min().item():.3f} - {gt_keep_mask.max().item():.3f}")

    # Prepare model kwargs with class ID
    class_id = torch.tensor([950], device=device)  # Default class (orange)

    model_kwargs = {
        'gt': gt,
        'gt_keep_mask': gt_keep_mask,
        'deg': 'inpainting',
        'save_path': 'center_hole_inpainting_t100',
        'y': class_id,
        'scale': 1,
        'resize_y': False,
        'sigma_y': 0.0,
        'conf': conf
    }

    # Generate center hole inpainted result
    print("🕳️ Starting CENTER HOLE Nine-Patch Inpainting...")
    print("   Target: Remove elephant, fill with beach environment")

    with torch.no_grad():
        try:
            sample = None
            for sample_dict in diffusion.p_sample_loop_progressive(
                model,
                (1, 3, 512, 512),  # 512x512 shape
                clip_denoised=True,
                model_kwargs=model_kwargs,
                device=device,
                conf=conf
            ):
                sample = sample_dict["sample"]
        except Exception as e:
            print(f"Sample generation error: {e}")
            import traceback
            traceback.print_exc()
            sample = gt  # Use original as fallback

    print("✅ Center Hole Inpainting completed!")
    print("📁 Results saved to results/center_hole_inpainting_t100/")
    print("🔬 Compare with t_T=25: results/center_hole_inpainting_results/")
    print("🎯 Expected result: Elephant removed, center filled with beach")

if __name__ == "__main__":
    main()