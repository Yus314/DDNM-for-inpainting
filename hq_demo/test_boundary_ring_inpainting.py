#!/usr/bin/env python3
"""
境界リングインペインティングテスト
中央168x168領域の境界5pxのみを補完する

使用方法:
1. 元画像と他モデルで補完済みの中央領域を準備
2. このスクリプトで境界5pxのみを滑らかに補完
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
from utils.mask_generator import (
    create_boundary_ring_mask,
    visualize_mask_info,
    save_mask_as_image,
    composite_center_region
)


def load_and_prepare_images(
    original_path: str,
    inpainted_center_path: str = None,
    center_region_size: int = 168,
    image_size: int = 512,
    device: str = 'cpu'
):
    """
    元画像と補完済み中央領域を読み込み、合成画像を作成
    
    Args:
        original_path: 元の512x512画像のパス
        inpainted_center_path: 他モデルで補完済みの中央領域のパス (None の場合は元画像をそのまま使用)
        center_region_size: 中央領域サイズ
        image_size: 画像サイズ
        device: デバイス
    
    Returns:
        gt: 合成済みGT画像 [1, 3, H, W]
        original: 元画像 [1, 3, H, W]
    """
    # 元画像読み込み
    pil_original = Image.open(original_path).convert('RGB')
    if pil_original.size != (image_size, image_size):
        pil_original = pil_original.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    arr_original = np.array(pil_original).astype(np.float32) / 127.5 - 1  # [-1, 1]
    original = torch.from_numpy(np.transpose(arr_original, [2, 0, 1])).unsqueeze(0).to(device)
    
    if inpainted_center_path is not None and os.path.exists(inpainted_center_path):
        # 補完済み中央領域を読み込み
        pil_center = Image.open(inpainted_center_path).convert('RGB')
        
        # 中央領域サイズにリサイズ
        if pil_center.size != (center_region_size, center_region_size):
            pil_center = pil_center.resize((center_region_size, center_region_size), Image.Resampling.LANCZOS)
        
        arr_center = np.array(pil_center).astype(np.float32) / 127.5 - 1
        center = torch.from_numpy(np.transpose(arr_center, [2, 0, 1])).unsqueeze(0).to(device)
        
        # 合成
        gt = composite_center_region(original, center, center_region_size)
        print(f"✅ Composited center region from: {inpainted_center_path}")
    else:
        # 補完済み中央がない場合は元画像をそのまま使用
        gt = original.clone()
        print(f"⚠️ No inpainted center provided, using original image")
    
    return gt, original


def main():
    parser = argparse.ArgumentParser(description='Boundary Ring Inpainting Test')
    parser.add_argument('--original', type=str, 
                        default='data/datasets/gts/BSDS500_168_BLIP_22-44_11/100007.jpg',
                        help='Path to original 512x512 image')
    parser.add_argument('--inpainted_center', type=str, default=None,
                        help='Path to inpainted center region (168x168)')
    parser.add_argument('--center_size', type=int, default=168,
                        help='Center region size')
    parser.add_argument('--boundary_width', type=int, default=5,
                        help='Boundary ring width to inpaint')
    parser.add_argument('--output_dir', type=str, default='boundary_ring_inpainting_results',
                        help='Output directory name')
    args = parser.parse_args()
    
    # Load configuration
    config_path = "confs/inpainting_512_nine_patch_t100_uncond.yml"
    conf = conf_base.Default_Conf()
    conf.update(yamlread(config_path))
    
    print(f"📋 Using config: {config_path}")
    print(f"   t_T = {conf.schedule_jump_params['t_T']}")
    print(f"   class_cond = {conf.class_cond}")
    print(f"\n🎯 Boundary Ring Inpainting Settings:")
    print(f"   Center region: {args.center_size}x{args.center_size}")
    print(f"   Boundary width: {args.boundary_width}px")

    # Setup device
    device = dist_util.dev(conf.get('device'))

    # Create model and diffusion
    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )

    # Load pretrained model
    print(f"\nLoading model from {conf.model_path}")
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # Load and prepare images
    print(f"\n📷 Loading images...")
    gt, original = load_and_prepare_images(
        original_path=args.original,
        inpainted_center_path=args.inpainted_center,
        center_region_size=args.center_size,
        image_size=512,
        device=device
    )

    # Create boundary ring mask
    print(f"\n🎭 Creating boundary ring mask...")
    gt_keep_mask = create_boundary_ring_mask(
        image_size=512,
        center_region_size=args.center_size,
        boundary_width=args.boundary_width,
        device=device
    )
    
    visualize_mask_info(gt_keep_mask, f"Boundary Ring Mask ({args.center_size}x{args.center_size}, {args.boundary_width}px)")

    # Save mask for visualization
    mask_save_path = f"data/datasets/gt_keep_masks/boundary_ring_{args.center_size}_{args.boundary_width}px/mask.png"
    save_mask_as_image(gt_keep_mask, mask_save_path)

    print(f"\nGT tensor shape: {gt.shape}")
    print(f"Mask tensor shape: {gt_keep_mask.shape}")

    # Model kwargs
    model_kwargs = {
        'gt': gt,
        'gt_keep_mask': gt_keep_mask,
        'deg': 'inpainting',
        'save_path': args.output_dir,
        'y': None,  # Unconditional
        'scale': 1,
        'resize_y': False,
        'sigma_y': 0.0,
        'conf': conf
    }

    # Generate
    print(f"\n🔧 Starting Boundary Ring Inpainting...")
    print(f"   Only {args.boundary_width}px boundary will be inpainted")

    with torch.no_grad():
        try:
            sample = None
            for sample_dict in diffusion.p_sample_loop_progressive(
                model,
                (1, 3, 512, 512),
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
            sample = gt

    print(f"\n✅ Boundary Ring Inpainting completed!")
    print(f"📁 Results saved to results/{args.output_dir}/")


if __name__ == "__main__":
    main()
