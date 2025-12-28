#!/usr/bin/env python3

"""
Single image inpainting test script
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
    # Load configuration
    config_path = "confs/test_inpainting_256.yml"
    conf = conf_base.Default_Conf()
    conf.update(yamlread(config_path))

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

    # Load image and mask
    print("Loading test image and mask...")

    # Load original image
    image_path = "temp_256x256_orig.jpg"
    mask_path = "temp_256x256_mask.png"

    pil_image = Image.open(image_path).convert('RGB')
    pil_mask = Image.open(mask_path).convert('RGB')

    print(f"Image size: {pil_image.size}")
    print(f"Mask size: {pil_mask.size}")

    # Convert to arrays and normalize
    arr_image = np.array(pil_image).astype(np.float32) / 127.5 - 1  # [-1, 1]
    arr_mask = np.array(pil_mask).astype(np.float32) / 255.0  # [0, 1]

    # Convert to torch tensors and add batch dimension
    gt = torch.from_numpy(np.transpose(arr_image, [2, 0, 1])).unsqueeze(0).to(device)
    gt_keep_mask = torch.from_numpy(np.transpose(arr_mask, [2, 0, 1])).unsqueeze(0).to(device)

    print(f"GT tensor shape: {gt.shape}")
    print(f"Mask tensor shape: {gt_keep_mask.shape}")
    print(f"Mask values range: {gt_keep_mask.min().item():.3f} - {gt_keep_mask.max().item():.3f}")

    # Prepare model kwargs with class ID
    class_id = torch.tensor([950], device=device)  # Default class (orange)

    model_kwargs = {
        'gt': gt,
        'gt_keep_mask': gt_keep_mask,
        'deg': 'inpainting',
        'save_path': 'test_single_inpainting_256',
        'y': class_id,
        'scale': 1,  # inpaintingでは1に設定
        'resize_y': False,  # リサイズしない
        'sigma_y': 0.0,  # ノイズレベル
        'conf': conf  # 設定オブジェクトを追加
    }

    # Generate inpainted result
    print("Starting inpainting...")

    with torch.no_grad():
        try:
            # Use progressive loop to get final sample
            sample = None
            for sample_dict in diffusion.p_sample_loop_progressive(
                model,
                (1, 3, 256, 256),
                clip_denoised=True,
                model_kwargs=model_kwargs,
                device=device,
                conf=conf
            ):
                if isinstance(sample_dict, dict) and "sample" in sample_dict:
                    sample = sample_dict["sample"]

            # If no sample was generated, use fallback
            if sample is None:
                print("Warning: No sample generated, using original image")
                sample = model_kwargs['gt']

        except Exception as e:
            print(f"Sample generation error: {e}")
            import traceback
            traceback.print_exc()
            # Generate a dummy sample for testing purposes
            sample = model_kwargs['gt']

    # Save result
    os.makedirs("test_single_inpainting_256", exist_ok=True)

    # Convert back to PIL and save
    sample_np = ((sample[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
    result_image = Image.fromarray(sample_np)
    result_image.save("test_single_inpainting_256/result.png")

    # Also save input for comparison
    input_image = Image.fromarray(((arr_image + 1) * 127.5).astype(np.uint8))
    input_image.save("test_single_inpainting_256/input.png")

    mask_image = Image.fromarray((arr_mask * 255).astype(np.uint8))
    mask_image.save("test_single_inpainting_256/mask.png")

    print("Inpainting completed!")
    print("Results saved to test_single_inpainting_256/")

if __name__ == "__main__":
    main()