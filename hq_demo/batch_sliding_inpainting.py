#!/usr/bin/env python3
"""
指定ディレクトリに対するスライディングウィンドウインペイントバッチ処理スクリプト

Usage:
    python batch_sliding_inpainting.py \
        --input_dir <画像ディレクトリ> \
        --mask_type <center_mask|boundary_ring_5px|boundary_ring_10px> \
        --output_dir <出力ディレクトリ> \
        [--config <設定ファイル>] \
        [--start <開始インデックス>] \
        [--limit <処理枚数制限>]

Features:
- 512x512画像のスライディングウィンドウインペイント（自動的に3x3=9ウィンドウ）
- 256x256画像も対応（1x1=1ウィンドウ）
- プログレッシブマスク更新による高品質な補完
- 全最終結果を集約したディレクトリ（final_results/）を自動作成
- エラーハンドリングと進行状況表示
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from conf_mgt import conf_base
from guided_diffusion import dist_util
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, select_args
from utils import (
    yamlread,
    setup_model,
    load_image_as_tensor,
    load_mask_as_tensor,
    tensor_to_pil,
    get_mask_path,
    setup_directories,
    get_image_list,
    write_log,
    create_batch_log_file,
)


# ============================================================================
# Utility functions now imported from utils module
# ============================================================================
# - get_mask_path: マスクタイプからファイルパスを取得
# - setup_directories: 出力ディレクトリの準備
# - get_image_list: 入力画像のリスト取得
# - write_log: ログファイルへの記録
# - setup_model: モデルと拡散プロセスのセットアップ
# - load_image_as_tensor: PIL画像をテンソルに変換
# - load_mask_as_tensor: マスクをテンソルに変換
# - tensor_to_pil: テンソルをPIL画像に変換
# ============================================================================


# ============================================================================
# Local model setup wrapper for sliding window mode
# ============================================================================

def setup_model_sliding(config_path: str = "confs/test_inpainting_256.yml"):
    """
    スライディングウィンドウモード用のモデルセットアップ

    Args:
        config_path: 設定ファイルパス

    Returns:
        (model, diffusion, conf, device) のタプル
    """
    model, diffusion, conf, device = setup_model(config_path)

    # Nine-Patch Modeを有効化（512x512画像の9分割処理用）
    conf.nine_patch_mode = True

    return model, diffusion, conf, device


# ============================================================================
# パッチ可視化関数
# ============================================================================

def create_patch_grid_visualization(output_dir: Path, image_name: str) -> Optional[Path]:
    """
    9つのパッチを3x3グリッドで表示した画像を作成

    Args:
        output_dir: 出力ディレクトリ
        image_name: 画像名（拡張子なし）

    Returns:
        作成した画像のパス、またはNone（失敗時）
    """
    # パッチディレクトリを探す
    patch_base = Path("results") / "results" / output_dir.name / image_name / "patches"

    if not patch_base.exists():
        print(f"  ⚠️  Patch directory not found: {patch_base}")
        return None

    # 各パッチの最終結果を読み込む
    patches = []
    for i in range(9):
        patch_dir = patch_base / f"patch_{i:02d}"
        # 最終結果（step_000.png）を探す
        result_path = patch_dir / "progressive" / "step_000.png"

        if result_path.exists():
            patch_img = Image.open(result_path)
            patches.append(patch_img)
        else:
            print(f"  ⚠️  Patch {i} result not found: {result_path}")
            # ダミー画像（グレー）
            patches.append(Image.new('RGB', (256, 256), color=(128, 128, 128)))

    # 3x3グリッドを作成
    grid_width = 256 * 3
    grid_height = 256 * 3
    grid = Image.new('RGB', (grid_width, grid_height))

    for i, patch_img in enumerate(patches):
        row = i // 3
        col = i % 3
        x = col * 256
        y = row * 256
        grid.paste(patch_img, (x, y))

    # グリッド線を描画
    draw = ImageDraw.Draw(grid)
    # 縦線
    for i in range(1, 3):
        x = i * 256
        draw.line([(x, 0), (x, grid_height)], fill=(255, 0, 0), width=2)
    # 横線
    for i in range(1, 3):
        y = i * 256
        draw.line([(0, y), (grid_width, y)], fill=(255, 0, 0), width=2)

    # パッチ番号を描画
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    for i in range(9):
        row = i // 3
        col = i % 3
        x = col * 256 + 10
        y = row * 256 + 10
        # 背景（黒）
        draw.rectangle([(x-2, y-2), (x+40, y+28)], fill=(0, 0, 0))
        # テキスト（白）
        draw.text((x, y), f"P{i}", fill=(255, 255, 255), font=font)

    # 保存
    output_path = output_dir / image_name / "patch_grid.png"
    grid.save(output_path)
    print(f"  📊 Patch grid saved: {output_path}")

    return output_path


def create_boundary_visualization(output_dir: Path, image_name: str, final_result: Image.Image) -> Optional[Path]:
    """
    最終結果にパッチの境界線を重ねて表示

    Args:
        output_dir: 出力ディレクトリ
        image_name: 画像名（拡張子なし）
        final_result: 最終結果画像

    Returns:
        作成した画像のパス、またはNone（失敗時）
    """
    # 画像をコピー
    viz = final_result.copy()
    draw = ImageDraw.Draw(viz)

    width, height = viz.size

    if width == 512 and height == 512:
        # 512x512の場合、3x3グリッド（各256x256、128pxオーバーラップ）
        # ウィンドウ位置:
        # (0,0)-(256,256), (0,128)-(256,384), (0,256)-(256,512)
        # (128,0)-(384,256), (128,128)-(384,384), (128,256)-(384,512)
        # (256,0)-(512,256), (256,128)-(512,384), (256,256)-(512,512)

        # 各ウィンドウの境界を描画
        window_coords = [
            (0, 0, 256, 256),      # P0
            (0, 128, 256, 384),    # P1
            (0, 256, 256, 512),    # P2
            (128, 0, 384, 256),    # P3
            (128, 128, 384, 384),  # P4
            (128, 256, 384, 512),  # P5
            (256, 0, 512, 256),    # P6
            (256, 128, 512, 384),  # P7
            (256, 256, 512, 512),  # P8
        ]

        colors = [
            (255, 0, 0),    # P0: 赤
            (0, 255, 0),    # P1: 緑
            (0, 0, 255),    # P2: 青
            (255, 255, 0),  # P3: 黄
            (255, 0, 255),  # P4: マゼンタ
            (0, 255, 255),  # P5: シアン
            (255, 128, 0),  # P6: オレンジ
            (128, 0, 255),  # P7: 紫
            (0, 255, 128),  # P8: 緑青
        ]

        for i, (x1, y1, x2, y2) in enumerate(window_coords):
            color = colors[i]
            # 境界線を描画
            draw.rectangle([(x1, y1), (x2-1, y2-1)], outline=color, width=2)

            # パッチ番号を描画
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                font = ImageFont.load_default()

            # 左上に番号を描画
            text_x = x1 + 5
            text_y = y1 + 5
            # 背景
            draw.rectangle([(text_x-2, text_y-2), (text_x+25, text_y+18)], fill=(0, 0, 0))
            # テキスト
            draw.text((text_x, text_y), f"P{i}", fill=color, font=font)

    # 保存
    output_path = output_dir / image_name / "boundary_visualization.png"
    viz.save(output_path)
    print(f"  🎨 Boundary visualization saved: {output_path}")

    return output_path


# ============================================================================
# スライディングウィンドウインペインティング処理
# ============================================================================

def process_single_image_sliding(
    model, diffusion, conf, device,
    image_path: Path,
    mask_path: str,
    output_base: Path,
    final_results_dir: Path
) -> Tuple[bool, float, Optional[str]]:
    """
    単一画像のスライディングウィンドウインペインティング処理

    512x512画像 → 自動的に3x3=9ウィンドウで処理
    256x256画像 → 1x1=1ウィンドウで処理

    Args:
        model, diffusion, conf, device: モデル関連
        image_path: 入力画像パス
        mask_path: マスク画像パス
        output_base: 出力ベースディレクトリ
        final_results_dir: 最終結果集約ディレクトリ

    Returns:
        (success, duration, error_message) のタプル
    """
    image_name = image_path.stem
    output_dir = output_base / image_name
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        print(f"\n🎯 Processing: {image_path.name}")

        # 画像とマスクの読み込み
        pil_image = Image.open(image_path).convert('RGB')
        pil_mask = Image.open(mask_path).convert('L')  # グレースケールに変換

        # サイズ確認とリサイズ
        print(f"  📐 Image size: {pil_image.size}, Mask size: {pil_mask.size}")

        if pil_image.size != pil_mask.size:
            # マスクを画像サイズにリサイズ
            print(f"  📐 Resizing mask from {pil_mask.size} to {pil_image.size}")
            pil_mask = pil_mask.resize(pil_image.size, Image.Resampling.LANCZOS)

        # テンソル変換
        arr_image = np.array(pil_image).astype(np.float32) / 127.5 - 1  # [-1, 1]
        arr_mask = np.array(pil_mask).astype(np.float32) / 255.0  # [0, 1] グレースケール（HxW）

        # マスクを反転（白=補完領域 → 黒=補完領域、白=既知領域）
        # gt_keep_mask: 1=既知領域（保持）、0=補完領域
        arr_mask = 1.0 - arr_mask

        gt = torch.from_numpy(np.transpose(arr_image, [2, 0, 1])).unsqueeze(0).to(device)
        # マスクを3チャンネルに拡張（RGBと同じ形状にする）
        arr_mask_3ch = np.stack([arr_mask, arr_mask, arr_mask], axis=0)  # [3, H, W]
        gt_keep_mask = torch.from_numpy(arr_mask_3ch).unsqueeze(0).to(device)

        print(f"  📊 Tensor: GT {gt.shape}, Mask {gt_keep_mask.shape}")
        print(f"  📊 Mask range: {gt_keep_mask.min().item():.3f} - {gt_keep_mask.max().item():.3f}")

        # model_kwargs設定
        class_id = torch.tensor([950], device=device)  # Default class (orange)

        model_kwargs = {
            'gt': gt,
            'gt_keep_mask': gt_keep_mask,
            'deg': 'inpainting',
            'save_path': str(output_dir),  # 個別画像のディレクトリ
            'y': class_id,
            'scale': 1,
            'resize_y': False,
            'sigma_y': 0.0,
            'conf': conf
        }

        # スライディングウィンドウインペインティング
        # p_sample_loop_progressiveが自動的にスライディング処理を実行
        # shapeは入力画像と同じサイズを渡す（512x512など）
        print(f"  🔄 Starting sliding window inpainting...")

        with torch.no_grad():
            sample = None
            for sample_dict in diffusion.p_sample_loop_progressive(
                model,
                gt.shape,  # 入力画像と同じサイズ（スライディング処理は内部で自動実行）
                clip_denoised=True,
                model_kwargs=model_kwargs,
                device=device,
                conf=conf
            ):
                sample = sample_dict["sample"]

        if sample is None:
            raise RuntimeError("No sample generated")

        # 結果保存
        print(f"  💾 Saving results...")

        result_image = tensor_to_pil(sample)

        # 1. final_results/に保存（集約ディレクトリ）
        final_result_path = final_results_dir / f"{image_name}_result.png"
        result_image.save(final_result_path)

        # 2. 比較用画像を保存
        pil_image.save(output_dir / "input.png")

        masked_input_image = tensor_to_pil(gt * gt_keep_mask)
        masked_input_image.save(output_dir / "masked_input.png")

        mask_image = Image.fromarray((arr_mask * 255).astype(np.uint8))
        mask_image.save(output_dir / "mask.png")

        # 3. パッチ可視化を作成
        print(f"  🎨 Creating patch visualizations...")
        create_patch_grid_visualization(output_dir.parent, image_name)
        create_boundary_visualization(output_dir.parent, image_name, result_image)

        duration = time.time() - start_time
        print(f"  ✅ Success: {image_name} ({duration:.1f}s)")

        return True, duration, None

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Error: {str(e)}"
        print(f"  ❌ Failed: {image_name} - {error_msg}")
        import traceback
        traceback.print_exc()
        return False, duration, error_msg


# ============================================================================
# 単一パッチ処理（スライディングウィンドウなし）
# ============================================================================

def process_single_image_single_pass(
    model, diffusion, conf, device,
    image_path: Path,
    mask_path: str,
    output_base: Path,
    final_results_dir: Path
) -> Tuple[bool, float, Optional[str]]:
    """
    単一パッチ処理: 中央256x256のみを切り出して1回で処理

    512x512画像から中央256x256を切り出し、境界リングマスクの対応領域を適用して
    1回のみ拡散処理を実行する。結果は元の512x512画像に合成される。

    Args:
        model, diffusion, conf, device: モデル関連
        image_path: 入力画像パス
        mask_path: マスク画像パス
        output_base: 出力ベースディレクトリ
        final_results_dir: 最終結果集約ディレクトリ

    Returns:
        (success, duration, error_message) のタプル
    """
    image_name = image_path.stem
    output_dir = output_base / image_name
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        print(f"\n🎯 Processing (Single Pass): {image_path.name}")

        # 画像とマスクの読み込み
        pil_image = Image.open(image_path).convert('RGB')
        pil_mask = Image.open(mask_path).convert('L')

        print(f"  📐 Original image size: {pil_image.size}, Mask size: {pil_mask.size}")

        # サイズ確認
        if pil_image.size != (512, 512):
            raise ValueError(f"Expected 512x512 image, got {pil_image.size}")
        if pil_mask.size != (512, 512):
            # マスクを512x512にリサイズ
            print(f"  📐 Resizing mask from {pil_mask.size} to (512, 512)")
            pil_mask = pil_mask.resize((512, 512), Image.Resampling.LANCZOS)

        # 中央256x256を切り出し [128:384, 128:384]
        crop_box = (128, 128, 384, 384)  # (left, upper, right, lower)
        cropped_image = pil_image.crop(crop_box)
        cropped_mask = pil_mask.crop(crop_box)

        print(f"  📐 Cropped center: {cropped_image.size}")

        # テンソル変換
        arr_image = np.array(cropped_image).astype(np.float32) / 127.5 - 1  # [-1, 1]
        arr_mask = np.array(cropped_mask).astype(np.float32) / 255.0  # [0, 1]

        # マスクを反転（白=補完領域 → gt_keep_mask: 1=既知領域、0=補完領域）
        arr_mask = 1.0 - arr_mask

        gt = torch.from_numpy(np.transpose(arr_image, [2, 0, 1])).unsqueeze(0).to(device)
        arr_mask_3ch = np.stack([arr_mask, arr_mask, arr_mask], axis=0)
        gt_keep_mask = torch.from_numpy(arr_mask_3ch).unsqueeze(0).to(device)

        print(f"  📊 Tensor: GT {gt.shape}, Mask {gt_keep_mask.shape}")
        print(f"  📊 Mask range: {gt_keep_mask.min().item():.3f} - {gt_keep_mask.max().item():.3f}")

        # nine_patch_mode を無効化して単一処理
        original_nine_patch_mode = getattr(conf, 'nine_patch_mode', False)
        conf.nine_patch_mode = False

        # model_kwargs設定
        class_id = torch.tensor([950], device=device)

        model_kwargs = {
            'gt': gt,
            'gt_keep_mask': gt_keep_mask,
            'deg': 'inpainting',
            'save_path': str(output_dir),
            'y': class_id,
            'scale': 1,
            'resize_y': False,
            'sigma_y': 0.0,
            'conf': conf
        }

        print(f"  🔄 Starting single pass inpainting (256x256 center only)...")

        with torch.no_grad():
            sample = None
            for sample_dict in diffusion.p_sample_loop_progressive(
                model,
                gt.shape,  # (1, 3, 256, 256)
                clip_denoised=True,
                model_kwargs=model_kwargs,
                device=device,
                conf=conf
            ):
                sample = sample_dict["sample"]

        # nine_patch_mode を元に戻す
        conf.nine_patch_mode = original_nine_patch_mode

        if sample is None:
            raise RuntimeError("No sample generated")

        # 結果保存
        print(f"  💾 Saving results...")

        result_256 = tensor_to_pil(sample)

        # 512x512に合成
        result_512 = pil_image.copy()
        result_512.paste(result_256, (128, 128))

        # 1. final_results/に保存
        final_result_path = final_results_dir / f"{image_name}_result.png"
        result_512.save(final_result_path)

        # 2. 比較用画像を保存
        pil_image.save(output_dir / "input.png")
        cropped_image.save(output_dir / "cropped_input.png")

        masked_cropped = tensor_to_pil(gt * gt_keep_mask)
        masked_cropped.save(output_dir / "masked_cropped_input.png")

        cropped_mask.save(output_dir / "cropped_mask.png")
        result_256.save(output_dir / "result_256.png")
        result_512.save(output_dir / "result_512.png")

        duration = time.time() - start_time
        print(f"  ✅ Success: {image_name} ({duration:.1f}s)")

        return True, duration, None

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Error: {str(e)}"
        print(f"  ❌ Failed: {image_name} - {error_msg}")
        import traceback
        traceback.print_exc()
        return False, duration, error_msg


# ============================================================================
# サマリーファイル作成
# ============================================================================

def create_summary_file(
    output_base: Path,
    args,
    total_images: int,
    success_count: int,
    failed_count: int,
    total_elapsed: float,
    total_processing_time: float
):
    """バッチ処理のサマリーファイルを作成"""
    summary_file = output_base / "summary.txt"
    with open(summary_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("Sliding Window Inpainting Batch Processing Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input directory: {args.input_dir}\n")
        f.write(f"Mask type: {args.mask_type}\n")
        f.write(f"Config file: {args.config}\n\n")
        f.write(f"Total images processed: {total_images}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {failed_count}\n")
        f.write(f"Total time: {total_elapsed/60:.1f} minutes\n")
        if success_count > 0:
            f.write(f"Average time: {total_processing_time/success_count:.1f}s per image\n")
        f.write(f"\nOutput directory: {args.output_dir}\n")
        f.write(f"Final results directory: {output_base / 'final_results'}\n")


# ============================================================================
# バッチ処理メイン
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sliding Window Inpainting Batch Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images with center mask
  python batch_sliding_inpainting.py \\
    --input_dir data/datasets/gts/BSDS500_168_BLIP_22-44_11 \\
    --mask_type center_mask \\
    --output_dir results/batch_sliding_center

  # Process first 3 images for testing
  python batch_sliding_inpainting.py \\
    --input_dir data/datasets/gts/BSDS500_168_BLIP_22-44_11 \\
    --mask_type boundary_ring_5px \\
    --output_dir results/test_sliding \\
    --limit 3
        """
    )

    parser.add_argument("--input_dir", required=True,
                       help="Input directory containing images")
    parser.add_argument("--mask_type", required=True,
                       choices=['center_mask', 'boundary_ring_5px', 'boundary_ring_10px', 'boundary_ring_136_10px'],
                       help="Mask type to use")
    parser.add_argument("--output_dir", required=True,
                       help="Output base directory")
    parser.add_argument("--config", default="confs/inpainting_512_nine_patch_t100_uncond.yml",
                       help="Configuration file path (default: confs/inpainting_512_nine_patch_t100_uncond.yml)")
    parser.add_argument("--start", type=int, default=0,
                       help="Start index (for resuming)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Maximum number of images to process")
    parser.add_argument("--single_pass", action="store_true",
                       help="中央256x256のみを1回で処理（スライディングウィンドウなし）")

    args = parser.parse_args()

    try:
        # セットアップ
        print("\n" + "="*80)
        if args.single_pass:
            print("🚀 Single Pass Inpainting Batch Processing (Center 256x256 only)")
        else:
            print("🚀 Sliding Window Inpainting Batch Processing")
        print("="*80)

        output_base, log_dir, final_results_dir = setup_directories(args.output_dir)
        image_files = get_image_list(args.input_dir)
        mask_path = get_mask_path(args.mask_type)

        # 処理範囲の決定
        if args.limit:
            end_idx = min(args.start + args.limit, len(image_files))
        else:
            end_idx = len(image_files)

        process_images = image_files[args.start:end_idx]

        # ログファイル
        log_file = create_batch_log_file(log_dir)

        print(f"\n📋 Configuration:")
        print(f"   Input directory: {args.input_dir}")
        print(f"   Mask type: {args.mask_type}")
        print(f"   Mask path: {mask_path}")
        print(f"   Output directory: {args.output_dir}")
        print(f"   Final results dir: {final_results_dir}")
        print(f"   Images to process: {len(process_images)} ({args.start} to {end_idx-1})")
        print(f"   Config file: {args.config}")
        print(f"   Processing mode: {'Single Pass (center 256x256)' if args.single_pass else 'Sliding Window'}")
        print(f"   Log file: {log_file}")

        # モデルセットアップ（1回のみ）
        print(f"\n🔧 Setting up model...")
        model, diffusion, conf, device = setup_model_sliding(args.config)

        # バッチ処理実行
        print(f"\n{'='*80}")
        print("🎬 Starting batch processing...")
        print(f"{'='*80}")

        total_start = time.time()
        success_count = 0
        failed_count = 0
        total_processing_time = 0

        write_log(log_file, f"Batch processing started: {len(process_images)} images")
        write_log(log_file, f"Mask type: {args.mask_type}")
        write_log(log_file, f"Config: {args.config}")

        for i, image_path in enumerate(process_images):
            current_idx = args.start + i
            print(f"\n{'='*80}")
            print(f"📸 [{current_idx+1}/{len(image_files)}] {image_path.name}")
            print(f"{'='*80}")

            if args.single_pass:
                success, duration, error = process_single_image_single_pass(
                    model, diffusion, conf, device,
                    image_path, mask_path, output_base, final_results_dir
                )
            else:
                success, duration, error = process_single_image_sliding(
                    model, diffusion, conf, device,
                    image_path, mask_path, output_base, final_results_dir
                )

            if success:
                success_count += 1
                total_processing_time += duration
                log_msg = f"SUCCESS: {image_path.name} ({duration:.1f}s)"
            else:
                failed_count += 1
                log_msg = f"FAILED: {image_path.name} - {error}"

            write_log(log_file, log_msg)

            # 進行状況サマリー
            elapsed = time.time() - total_start
            avg_time = total_processing_time / success_count if success_count > 0 else duration
            remaining = len(process_images) - i - 1
            eta = remaining * avg_time

            print(f"\n📊 Progress Summary:")
            print(f"   Completed: {i+1}/{len(process_images)}")
            print(f"   Success: {success_count} | Failed: {failed_count}")
            print(f"   Avg time: {avg_time:.1f}s per image")
            print(f"   ETA: {eta/60:.1f} minutes")

        # 最終結果
        total_elapsed = time.time() - total_start

        print(f"\n{'='*80}")
        print("🏁 Batch Processing Completed!")
        print(f"{'='*80}")
        print(f"   Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
        print(f"   Successful: {success_count}/{len(process_images)}")
        print(f"   Failed: {failed_count}/{len(process_images)}")
        if success_count > 0:
            print(f"   Average time: {total_processing_time/success_count:.1f}s per image")
        print(f"\n📁 Results:")
        print(f"   Individual results: {args.output_dir}/")
        print(f"   Final results (all): {final_results_dir}/")
        print(f"   Log file: {log_file}")

        # サマリーファイル作成
        create_summary_file(
            output_base, args,
            len(process_images), success_count, failed_count,
            total_elapsed, total_processing_time
        )

        summary_file = output_base / "summary.txt"
        print(f"   Summary file: {summary_file}")

        summary = (f"BATCH COMPLETE: {success_count} success, {failed_count} failed, "
                  f"{total_elapsed/60:.1f} minutes total")
        write_log(log_file, summary)

        print(f"\n✨ All done!\n")

    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
