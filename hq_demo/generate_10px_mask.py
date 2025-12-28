#!/usr/bin/env python3
"""
境界幅10pxのマスクを生成するスクリプト
"""
import torch
import numpy as np
from PIL import Image
import os


def create_boundary_ring_mask(
    image_size: int = 512,
    center_region_size: int = 168,
    boundary_width: int = 10,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    中央領域の境界リングのみを補完対象とするマスクを生成

    Args:
        image_size: 画像サイズ (512)
        center_region_size: 中央領域のサイズ (168)
        boundary_width: 境界幅 (10)
        device: デバイス

    Returns:
        mask: [1, 3, H, W], 1=保持, 0=補完対象
    """
    mask = torch.ones(1, 3, image_size, image_size, device=device)

    # 中央領域の座標
    start = (image_size - center_region_size) // 2  # 172
    end = start + center_region_size                 # 340

    # 内側領域の座標（境界を除く）
    inner_start = start + boundary_width             # 182
    inner_end = end - boundary_width                 # 330

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


def save_mask_as_image(mask: torch.Tensor, save_path: str):
    """
    マスクテンソルをPNG画像として保存

    Args:
        mask: [1, 3, H, W] テンソル
        save_path: 保存パス
    """
    mask_np = mask[0, 0].cpu().numpy()

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


if __name__ == "__main__":
    print("=== Generating Boundary Ring Mask with 10px border ===")

    # 境界幅10pxのマスクを生成
    mask = create_boundary_ring_mask(
        image_size=512,
        center_region_size=168,
        boundary_width=10
    )

    visualize_mask_info(mask, "Boundary Ring Mask (168x168, 10px border)")

    # 座標確認
    start = (512 - 168) // 2  # 172
    end = start + 168          # 340
    inner_start = start + 10   # 182
    inner_end = end - 10       # 330

    print(f"\n=== Coordinates ===")
    print(f"  Center region: [{start}:{end}, {start}:{end}]")
    print(f"  Inner region (kept): [{inner_start}:{inner_end}, {inner_start}:{inner_end}]")
    print(f"  Boundary ring: 10px wide")

    # 境界リングのピクセル数を計算
    # 168x168 - 148x148 = 28224 - 21904 = 6320 pixels
    outer_area = 168 * 168
    inner_area = 148 * 148
    boundary_area = outer_area - inner_area
    print(f"\n=== Boundary Ring Area ===")
    print(f"  Outer (168x168): {outer_area:,} pixels")
    print(f"  Inner (148x148): {inner_area:,} pixels")
    print(f"  Boundary ring: {boundary_area:,} pixels")

    # 新しい名前で保存
    save_path = "data/datasets/gt_keep_masks/boundary_ring_168_10px/mask.png"
    save_mask_as_image(mask, save_path)

    print("\n✅ 10px boundary ring mask generated successfully!")
