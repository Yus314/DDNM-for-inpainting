#!/usr/bin/env python3
"""
Mask Generator Utilities for Inpainting
境界リングマスク生成ユーティリティ
"""

import torch
import numpy as np
from PIL import Image
import os


def create_boundary_ring_mask(
    image_size: int = 512,
    center_region_size: int = 168,
    boundary_width: int = 5,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    中央領域の境界リングのみを補完対象とするマスクを生成
    
    Args:
        image_size: 画像サイズ (512)
        center_region_size: 中央領域のサイズ (168)
        boundary_width: 境界幅 (5)
        device: デバイス
    
    Returns:
        mask: [1, 3, H, W], 1=保持, 0=補完対象
    """
    mask = torch.ones(1, 3, image_size, image_size, device=device)
    
    # 中央領域の座標
    start = (image_size - center_region_size) // 2  # 172
    end = start + center_region_size                 # 340
    
    # 内側領域の座標（境界を除く）
    inner_start = start + boundary_width             # 177
    inner_end = end - boundary_width                 # 335
    
    # 境界リングを0に設定（4辺）
    # 上辺
    mask[:, :, start:inner_start, start:end] = 0
    # 下辺
    mask[:, :, inner_end:end, start:end] = 0
    # 左辺
    mask[:, :, inner_start:inner_end, start:inner_start] = 0
    # 右辺
    mask[:, :, inner_start:inner_end, inner_end:end] = 0
    
    return mask


def create_center_hole_mask(
    image_size: int = 512,
    hole_size: int = 168,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    中央に正方形の穴があるマスクを生成
    
    Args:
        image_size: 画像サイズ
        hole_size: 穴のサイズ
        device: デバイス
    
    Returns:
        mask: [1, 3, H, W], 1=保持, 0=補完対象
    """
    mask = torch.ones(1, 3, image_size, image_size, device=device)
    
    start = (image_size - hole_size) // 2
    end = start + hole_size
    
    mask[:, :, start:end, start:end] = 0
    
    return mask


def save_mask_as_image(mask: torch.Tensor, save_path: str):
    """
    マスクテンソルをPNG画像として保存
    
    Args:
        mask: [1, 3, H, W] or [H, W] テンソル
        save_path: 保存パス
    """
    if mask.dim() == 4:
        mask_np = mask[0, 0].cpu().numpy()
    elif mask.dim() == 2:
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask[0].cpu().numpy()
    
    # 0-255にスケール
    mask_img = (mask_np * 255).astype(np.uint8)
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(mask_img).save(save_path)
    print(f"Mask saved to: {save_path}")


def visualize_mask_info(mask: torch.Tensor, name: str = "Mask"):
    """
    マスクの情報を表示
    
    Args:
        mask: マスクテンソル
        name: 表示名
    """
    total_pixels = mask.numel() // 3  # 3チャンネル分を除く
    keep_pixels = (mask > 0.5).sum().item() // 3
    inpaint_pixels = total_pixels - keep_pixels
    
    print(f"\n=== {name} Info ===")
    print(f"  Shape: {mask.shape}")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Keep pixels (mask=1): {keep_pixels:,} ({keep_pixels/total_pixels*100:.2f}%)")
    print(f"  Inpaint pixels (mask=0): {inpaint_pixels:,} ({inpaint_pixels/total_pixels*100:.2f}%)")


def composite_center_region(
    original_img: torch.Tensor,
    inpainted_center: torch.Tensor,
    center_region_size: int = 168
) -> torch.Tensor:
    """
    元画像の中央領域に補完済み結果を合成
    
    Args:
        original_img: 元画像 [1, 3, H, W]
        inpainted_center: 補完済み中央領域 [1, 3, center_size, center_size]
        center_region_size: 中央領域サイズ
    
    Returns:
        合成画像 [1, 3, H, W]
    """
    result = original_img.clone()
    
    h, w = original_img.shape[2], original_img.shape[3]
    start_h = (h - center_region_size) // 2
    start_w = (w - center_region_size) // 2
    end_h = start_h + center_region_size
    end_w = start_w + center_region_size
    
    # 中央領域に補完結果を配置
    if inpainted_center.shape[2] != center_region_size or inpainted_center.shape[3] != center_region_size:
        # リサイズが必要な場合
        import torch.nn.functional as F
        inpainted_center = F.interpolate(
            inpainted_center, 
            size=(center_region_size, center_region_size), 
            mode='bilinear', 
            align_corners=False
        )
    
    result[:, :, start_h:end_h, start_w:end_w] = inpainted_center
    
    return result


# テスト用
if __name__ == "__main__":
    print("=== Boundary Ring Mask Test ===")
    
    # 境界リングマスク生成
    mask = create_boundary_ring_mask(
        image_size=512,
        center_region_size=168,
        boundary_width=5
    )
    
    visualize_mask_info(mask, "Boundary Ring Mask (168x168, 5px border)")
    
    # 座標確認
    start = (512 - 168) // 2  # 172
    end = start + 168          # 340
    inner_start = start + 5    # 177
    inner_end = end - 5        # 335
    
    print(f"\n=== Coordinates ===")
    print(f"  Center region: [{start}:{end}, {start}:{end}]")
    print(f"  Inner region (kept): [{inner_start}:{inner_end}, {inner_start}:{inner_end}]")
    print(f"  Boundary ring: 5px wide")
    
    # 境界リングのピクセル数を計算
    # 168x168 - 158x158 = 28224 - 24964 = 3260 pixels
    outer_area = 168 * 168
    inner_area = 158 * 158
    boundary_area = outer_area - inner_area
    print(f"\n=== Boundary Ring Area ===")
    print(f"  Outer (168x168): {outer_area:,} pixels")
    print(f"  Inner (158x158): {inner_area:,} pixels")
    print(f"  Boundary ring: {boundary_area:,} pixels")
    
    # マスク保存テスト
    save_mask_as_image(mask, "data/datasets/gt_keep_masks/boundary_ring_168_5px/mask.png")
    
    print("\n✅ Test completed!")
