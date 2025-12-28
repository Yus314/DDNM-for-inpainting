#!/usr/bin/env python3

"""
修正版: 512x512画像の9分割Inpainting システム
Nine-Patch Inpainting: 256x256モデルを使用した512x512画像の確実な補完
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from typing import List, Tuple

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from conf_mgt import conf_base
from guided_diffusion import dist_util
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, select_args
from utils import yamlread


def setup_model(config_path: str = "confs/test_inpainting_256.yml"):
    """256x256 inpaintingモデルのセットアップ"""
    conf = conf_base.Default_Conf()
    conf.update(yamlread(config_path))

    device = dist_util.dev(conf.get('device'))

    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )

    print(f"Loading model from {conf.model_path}")
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    return model, diffusion, conf, device


def inpaint_single_patch(model, diffusion, conf, device, patch_image: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
    """
    単一256x256パッチのインペインティング処理
    test_single_inpainting.py の成功パターンを使用
    """
    class_id = torch.tensor([950], device=device)  # Default class (orange)

    model_kwargs = {
        'gt': patch_image,
        'gt_keep_mask': patch_mask,
        'deg': 'inpainting',
        'save_path': 'temp_patch_inpainting',
        'y': class_id,
        'scale': 1,
        'resize_y': False,
        'sigma_y': 0.0,
        'conf': conf
    }

    with torch.no_grad():
        try:
            sample = None
            for sample_dict in diffusion.p_sample_loop_progressive(
                model,
                (1, 3, 256, 256),
                clip_denoised=True,
                model_kwargs=model_kwargs,
                device=device,
                conf=conf
            ):
                sample = sample_dict["sample"]
            return sample if sample is not None else patch_image
        except Exception as e:
            print(f"Patch inpainting error: {e}")
            return patch_image


def extract_patch(image: torch.Tensor, mask: torch.Tensor, h_start: int, h_end: int, w_start: int, w_end: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract a 256x256 patch from 512x512 image and mask"""
    patch_img = image[:, :, h_start:h_end, w_start:w_end]
    patch_mask = mask[:, :, h_start:h_end, w_start:w_end]

    # Ensure 256x256 size
    if patch_img.shape[2] != 256 or patch_img.shape[3] != 256:
        patch_img = F.interpolate(patch_img, size=(256, 256), mode='bilinear', align_corners=False)
        patch_mask = F.interpolate(patch_mask, size=(256, 256), mode='bilinear', align_corners=False)

    return patch_img, patch_mask


def blend_patches(patches: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    シンプルな重み付きブレンディング
    overlapping領域では平均値を取る
    """
    patch_coords = [
        (0, 256, 0, 256),     # セクション 1: 左上
        (0, 256, 128, 384),   # セクション 2: 上中央
        (0, 256, 256, 512),   # セクション 3: 右上
        (128, 384, 0, 256),   # セクション 4: 左中央
        (128, 384, 128, 384), # セクション 5: 中央
        (128, 384, 256, 512), # セクション 6: 右中央
        (256, 512, 0, 256),   # セクション 7: 左下
        (256, 512, 128, 384), # セクション 8: 下中央
        (256, 512, 256, 512), # セクション 9: 右下
    ]

    B, C = patches[0].shape[:2]
    result = torch.zeros(B, C, 512, 512, device=device)
    weight_map = torch.zeros(B, C, 512, 512, device=device)

    for i, patch in enumerate(patches):
        h_start, h_end, w_start, w_end = patch_coords[i]

        # Simple uniform weight for each patch
        patch_weight = torch.ones_like(patch)

        # Add weighted patch to result
        result[:, :, h_start:h_end, w_start:w_end] += patch * patch_weight
        weight_map[:, :, h_start:h_end, w_start:w_end] += patch_weight

    # Normalize by weight map to handle overlaps
    result = result / (weight_map + 1e-8)

    return result


def process_512x512_image(image_path: str, mask_path: str, output_dir: str) -> str:
    """
    512x512画像の九分割インペインティング処理
    """
    print(f"🎯 Starting Nine-Patch Inpainting: {image_path}")

    # Setup model
    model, diffusion, conf, device = setup_model()

    # Load and prepare images
    pil_image = Image.open(image_path).convert('RGB')
    if pil_image.size != (512, 512):
        print(f"Resizing image from {pil_image.size} to (512, 512)")
        pil_image = pil_image.resize((512, 512), Image.Resampling.LANCZOS)

    pil_mask = Image.open(mask_path).convert('RGB')
    if pil_mask.size != (512, 512):
        print(f"Resizing mask from {pil_mask.size} to (512, 512)")
        pil_mask = pil_mask.resize((512, 512), Image.Resampling.LANCZOS)

    # Convert to tensors
    arr_image = np.array(pil_image).astype(np.float32) / 127.5 - 1  # [-1, 1]
    arr_mask = np.array(pil_mask).astype(np.float32) / 255.0  # [0, 1]

    gt = torch.from_numpy(np.transpose(arr_image, [2, 0, 1])).unsqueeze(0).to(device)
    gt_keep_mask = torch.from_numpy(np.transpose(arr_mask, [2, 0, 1])).unsqueeze(0).to(device)

    print(f"📊 Input: GT {gt.shape}, Mask {gt_keep_mask.shape}")
    print(f"📊 Mask range: {gt_keep_mask.min().item():.3f} - {gt_keep_mask.max().item():.3f}")

    # 9分割の座標定義
    patch_coords = [
        (0, 256, 0, 256),     # セクション 1: 左上
        (0, 256, 128, 384),   # セクション 2: 上中央
        (0, 256, 256, 512),   # セクション 3: 右上
        (128, 384, 0, 256),   # セクション 4: 左中央
        (128, 384, 128, 384), # セクション 5: 中央
        (128, 384, 256, 512), # セクション 6: 右中央
        (256, 512, 0, 256),   # セクション 7: 左下
        (256, 512, 128, 384), # セクション 8: 下中央
        (256, 512, 256, 512), # セクション 9: 右下
    ]

    # Process each patch
    inpainted_patches = []
    print(f"🔄 Processing 9 patches...")

    for i, (h_start, h_end, w_start, w_end) in enumerate(patch_coords):
        print(f"  Patch {i+1}/9: [{h_start}:{h_end}, {w_start}:{w_end}]")

        # Extract patch
        patch_img, patch_mask = extract_patch(gt, gt_keep_mask, h_start, h_end, w_start, w_end)

        # Inpaint patch using successful pattern from test_single_inpainting.py
        inpainted_patch = inpaint_single_patch(model, diffusion, conf, device, patch_img, patch_mask)
        inpainted_patches.append(inpainted_patch)

    # Blend patches
    print(f"🎨 Blending patches...")
    final_result = blend_patches(inpainted_patches, device)

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save final result
    result_np = ((final_result[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
    result_np = np.clip(result_np, 0, 255)
    result_image = Image.fromarray(result_np)
    result_path = os.path.join(output_dir, "nine_patch_result_fixed.png")
    result_image.save(result_path)

    # Save comparison images
    input_image = Image.fromarray(((arr_image + 1) * 127.5).astype(np.uint8))
    input_image.save(os.path.join(output_dir, "input_fixed.png"))

    masked_input_np = ((gt * gt_keep_mask)[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5
    masked_input_np = np.clip(masked_input_np.astype(np.uint8), 0, 255)
    masked_input_image = Image.fromarray(masked_input_np)
    masked_input_image.save(os.path.join(output_dir, "masked_input_fixed.png"))

    mask_image = Image.fromarray((arr_mask * 255).astype(np.uint8))
    mask_image.save(os.path.join(output_dir, "mask_fixed.png"))

    print(f"✅ Nine-Patch Inpainting completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📄 Final result: {result_path}")

    return result_path


def main():
    """修正版九分割インペインティングのテスト実行"""

    # Process test image
    image_path = "data/datasets/gts/BSDS500_168_BLIP_22-44_11/100007.jpg"
    mask_path = "data/datasets/gt_keep_masks/center_mask_512/center_mask_512x512.png"
    output_dir = "nine_patch_results_fixed"

    # Perform nine-patch inpainting
    result_path = process_512x512_image(image_path, mask_path, output_dir)

    print(f"\n🎯 Fixed Nine-Patch Inpainting Test Completed!")
    print(f"📊 Method: 512x512 → 9×256x256 → inpainting → blended 512x512")
    print(f"🎨 Result: {result_path}")


if __name__ == "__main__":
    main()