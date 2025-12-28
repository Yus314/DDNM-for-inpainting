#!/usr/bin/env python3
"""
指定ディレクトリに対する9分割インペイントバッチ処理スクリプト

Usage:
    python batch_nine_patch_inpainting.py \
        --input_dir <画像ディレクトリ> \
        --mask_type <center_mask|boundary_ring_5px|boundary_ring_10px> \
        --output_dir <出力ディレクトリ> \
        [--config <設定ファイル>] \
        [--start <開始インデックス>] \
        [--limit <処理枚数制限>]

Features:
- 512x512画像の9分割インペイント（256x256モデル使用）
- 3種類のマスクタイプから選択可能
- 全最終結果を集約したディレクトリ（final_results/）を自動作成
- 中間パッチ結果も保存（デバッグ用）
- エラーハンドリングと進行状況表示
"""

import os
import sys
import glob
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

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


# ============================================================================
# マスク管理システム
# ============================================================================

MASK_PATHS = {
    'center_mask': 'data/datasets/gt_keep_masks/center_mask_512/center_mask_512x512.png',
    'boundary_ring_5px': 'data/datasets/gt_keep_masks/boundary_ring_168_5px/mask.png',
    'boundary_ring_10px': 'data/datasets/gt_keep_masks/boundary_ring_168_10px/mask.png',
}


def get_mask_path(mask_type: str) -> str:
    """
    マスクタイプから実際のファイルパスを返す

    Args:
        mask_type: マスクタイプ ('center_mask', 'boundary_ring_5px', 'boundary_ring_10px')

    Returns:
        マスクファイルの絶対パス

    Raises:
        ValueError: 不正なマスクタイプの場合
        FileNotFoundError: マスクファイルが存在しない場合
    """
    if mask_type not in MASK_PATHS:
        available = ', '.join(MASK_PATHS.keys())
        raise ValueError(f"Invalid mask_type: {mask_type}. Available: {available}")

    mask_path = MASK_PATHS[mask_type]

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    return mask_path


# ============================================================================
# ディレクトリ・ファイル管理
# ============================================================================

def setup_directories(base_output_dir: str) -> Tuple[Path, Path, Path]:
    """
    出力ディレクトリの準備

    Args:
        base_output_dir: ベース出力ディレクトリパス

    Returns:
        (base_path, log_dir, final_results_dir) のタプル
    """
    base_path = Path(base_output_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # ログディレクトリ
    log_dir = base_path / "logs"
    log_dir.mkdir(exist_ok=True)

    # 最終結果を集約するディレクトリ
    final_results_dir = base_path / "final_results"
    final_results_dir.mkdir(exist_ok=True)

    return base_path, log_dir, final_results_dir


def get_image_list(input_dir: str) -> List[Path]:
    """
    入力ディレクトリから画像ファイルのリストを取得

    Args:
        input_dir: 入力ディレクトリパス

    Returns:
        ソートされた画像ファイルパスのリスト

    Raises:
        FileNotFoundError: ディレクトリが存在しない場合
        ValueError: 画像ファイルが見つからない場合
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # サポートする画像拡張子
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(ext))

    # ソート
    image_files = sorted(image_files)

    if len(image_files) == 0:
        raise ValueError(f"No image files found in {input_dir}")

    print(f"Found {len(image_files)} images in {input_dir}")

    return image_files


def write_log(log_file: Path, message: str):
    """ログファイルに記録"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


# ============================================================================
# モデルセットアップ
# ============================================================================

def setup_model(config_path: str = "confs/test_inpainting_256.yml"):
    """
    256x256 inpaintingモデルのセットアップ

    Args:
        config_path: 設定ファイルパス

    Returns:
        (model, diffusion, conf, device) のタプル
    """
    print(f"Loading configuration from {config_path}")
    conf = conf_base.Default_Conf()
    conf.update(yamlread(config_path))

    device = dist_util.dev(conf.get('device'))

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )

    print(f"Loading model weights from {conf.model_path}")
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    print(f"✅ Model loaded successfully on {device}")

    return model, diffusion, conf, device


# ============================================================================
# 9分割インペイント処理
# ============================================================================

# 9分割の座標定義
PATCH_COORDS = [
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


def extract_patch(image: torch.Tensor, mask: torch.Tensor,
                 h_start: int, h_end: int, w_start: int, w_end: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    512x512画像から256x256パッチを抽出

    Args:
        image: 入力画像テンソル [B, C, 512, 512]
        mask: 入力マスクテンソル [B, C, 512, 512]
        h_start, h_end, w_start, w_end: パッチ座標

    Returns:
        (patch_image, patch_mask) のタプル、各 [B, C, 256, 256]
    """
    patch_img = image[:, :, h_start:h_end, w_start:w_end]
    patch_mask = mask[:, :, h_start:h_end, w_start:w_end]

    # 256x256サイズを保証
    if patch_img.shape[2] != 256 or patch_img.shape[3] != 256:
        patch_img = F.interpolate(patch_img, size=(256, 256), mode='bilinear', align_corners=False)
        patch_mask = F.interpolate(patch_mask, size=(256, 256), mode='bilinear', align_corners=False)

    return patch_img, patch_mask


def inpaint_single_patch(model, diffusion, conf, device,
                        patch_image: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
    """
    単一256x256パッチのインペインティング処理

    Args:
        model: 拡散モデル
        diffusion: 拡散プロセス
        conf: 設定オブジェクト
        device: デバイス
        patch_image: パッチ画像 [1, 3, 256, 256]
        patch_mask: パッチマスク [1, 3, 256, 256]

    Returns:
        インペイント済みパッチ [1, 3, 256, 256]
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
            print(f"  ⚠️  Patch inpainting error: {e}")
            return patch_image


def blend_patches(patches: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    9個のパッチを512x512画像にブレンド

    Args:
        patches: 9個のインペイント済みパッチのリスト
        device: デバイス

    Returns:
        ブレンド済み512x512画像 [1, 3, 512, 512]
    """
    B, C = patches[0].shape[:2]
    result = torch.zeros(B, C, 512, 512, device=device)
    weight_map = torch.zeros(B, C, 512, 512, device=device)

    for i, patch in enumerate(patches):
        h_start, h_end, w_start, w_end = PATCH_COORDS[i]

        # 均一な重み
        patch_weight = torch.ones_like(patch)

        # 重み付きパッチを結果に追加
        result[:, :, h_start:h_end, w_start:w_end] += patch * patch_weight
        weight_map[:, :, h_start:h_end, w_start:w_end] += patch_weight

    # オーバーラップ領域を重みで正規化
    result = result / (weight_map + 1e-8)

    return result


def save_patch_results(patches: List[torch.Tensor], output_dir: Path):
    """
    9個のパッチを個別に保存

    Args:
        patches: パッチのリスト
        output_dir: 出力ディレクトリ（patches/サブディレクトリに保存）
    """
    patch_dir = output_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    for i, patch in enumerate(patches):
        patch_np = ((patch[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
        patch_np = np.clip(patch_np, 0, 255)
        patch_image = Image.fromarray(patch_np)
        patch_path = patch_dir / f"patch_{i}.png"
        patch_image.save(patch_path)


def process_single_image_nine_patch(
    model, diffusion, conf, device,
    image_path: Path,
    mask_path: str,
    output_base: Path,
    final_results_dir: Path
) -> Tuple[bool, float, Optional[str]]:
    """
    単一画像の9分割インペイント処理

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
        if pil_image.size != (512, 512):
            print(f"  📐 Resizing image from {pil_image.size} to (512, 512)")
            pil_image = pil_image.resize((512, 512), Image.Resampling.LANCZOS)

        pil_mask = Image.open(mask_path).convert('RGB')
        if pil_mask.size != (512, 512):
            print(f"  📐 Resizing mask from {pil_mask.size} to (512, 512)")
            pil_mask = pil_mask.resize((512, 512), Image.Resampling.LANCZOS)

        # テンソル変換
        arr_image = np.array(pil_image).astype(np.float32) / 127.5 - 1  # [-1, 1]
        arr_mask = np.array(pil_mask).astype(np.float32) / 255.0  # [0, 1]

        gt = torch.from_numpy(np.transpose(arr_image, [2, 0, 1])).unsqueeze(0).to(device)
        gt_keep_mask = torch.from_numpy(np.transpose(arr_mask, [2, 0, 1])).unsqueeze(0).to(device)

        # 9分割処理
        inpainted_patches = []
        print(f"  🔄 Processing 9 patches...")

        for i, (h_start, h_end, w_start, w_end) in enumerate(PATCH_COORDS):
            print(f"    Patch {i+1}/9: [{h_start}:{h_end}, {w_start}:{w_end}]", end=" ")

            patch_start = time.time()

            # パッチ抽出
            patch_img, patch_mask = extract_patch(gt, gt_keep_mask, h_start, h_end, w_start, w_end)

            # インペイント
            inpainted_patch = inpaint_single_patch(model, diffusion, conf, device, patch_img, patch_mask)
            inpainted_patches.append(inpainted_patch)

            patch_duration = time.time() - patch_start
            print(f"✅ ({patch_duration:.1f}s)")

        # ブレンド
        print(f"  🎨 Blending patches...")
        final_result = blend_patches(inpainted_patches, device)

        # 結果保存
        print(f"  💾 Saving results...")

        # 1. 最終結果（個別ディレクトリ）
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        result_np = ((final_result[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
        result_np = np.clip(result_np, 0, 255)
        result_image = Image.fromarray(result_np)

        result_path = final_dir / "result.png"
        result_image.save(result_path)

        # 2. 最終結果（集約ディレクトリ）
        final_result_path = final_results_dir / f"{image_name}_result.png"
        result_image.save(final_result_path)

        # 3. パッチ結果
        save_patch_results(inpainted_patches, output_dir)

        # 4. 比較用画像
        input_image = Image.fromarray(((arr_image + 1) * 127.5).astype(np.uint8))
        input_image.save(output_dir / "input.png")

        masked_input_np = ((gt * gt_keep_mask)[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5
        masked_input_np = np.clip(masked_input_np.astype(np.uint8), 0, 255)
        masked_input_image = Image.fromarray(masked_input_np)
        masked_input_image.save(output_dir / "masked_input.png")

        mask_image = Image.fromarray((arr_mask * 255).astype(np.uint8))
        mask_image.save(output_dir / "mask.png")

        duration = time.time() - start_time
        print(f"  ✅ Success: {image_name} ({duration:.1f}s)")

        return True, duration, None

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Error: {str(e)}"
        print(f"  ❌ Failed: {image_name} - {error_msg}")
        return False, duration, error_msg


# ============================================================================
# バッチ処理メイン
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Nine-Patch Inpainting Batch Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images with center mask
  python batch_nine_patch_inpainting.py \\
    --input_dir data/datasets/gts/BSDS500_168_BLIP_22-44_11 \\
    --mask_type center_mask \\
    --output_dir results/batch_nine_patch_center

  # Process first 3 images for testing
  python batch_nine_patch_inpainting.py \\
    --input_dir data/datasets/gts/BSDS500_168_BLIP_22-44_11 \\
    --mask_type boundary_ring_5px \\
    --output_dir results/test_batch \\
    --limit 3
        """
    )

    parser.add_argument("--input_dir", required=True,
                       help="Input directory containing images")
    parser.add_argument("--mask_type", required=True,
                       choices=['center_mask', 'boundary_ring_5px', 'boundary_ring_10px'],
                       help="Mask type to use")
    parser.add_argument("--output_dir", required=True,
                       help="Output base directory")
    parser.add_argument("--config", default="confs/test_inpainting_256.yml",
                       help="Configuration file path (default: confs/test_inpainting_256.yml)")
    parser.add_argument("--start", type=int, default=0,
                       help="Start index (for resuming)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Maximum number of images to process")

    args = parser.parse_args()

    try:
        # セットアップ
        print("\n" + "="*80)
        print("🚀 Nine-Patch Inpainting Batch Processing")
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
        log_file = log_dir / f"batch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        print(f"\n📋 Configuration:")
        print(f"   Input directory: {args.input_dir}")
        print(f"   Mask type: {args.mask_type}")
        print(f"   Mask path: {mask_path}")
        print(f"   Output directory: {args.output_dir}")
        print(f"   Final results dir: {final_results_dir}")
        print(f"   Images to process: {len(process_images)} ({args.start} to {end_idx-1})")
        print(f"   Config file: {args.config}")
        print(f"   Log file: {log_file}")

        # モデルセットアップ（1回のみ）
        print(f"\n🔧 Setting up model...")
        model, diffusion, conf, device = setup_model(args.config)

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

            success, duration, error = process_single_image_nine_patch(
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
        summary_file = output_base / "summary.txt"
        with open(summary_file, "w") as f:
            f.write("="*80 + "\n")
            f.write("Nine-Patch Inpainting Batch Processing Summary\n")
            f.write("="*80 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input directory: {args.input_dir}\n")
            f.write(f"Mask type: {args.mask_type}\n")
            f.write(f"Config file: {args.config}\n\n")
            f.write(f"Total images processed: {len(process_images)}\n")
            f.write(f"Successful: {success_count}\n")
            f.write(f"Failed: {failed_count}\n")
            f.write(f"Total time: {total_elapsed/60:.1f} minutes\n")
            if success_count > 0:
                f.write(f"Average time: {total_processing_time/success_count:.1f}s per image\n")
            f.write(f"\nOutput directory: {args.output_dir}\n")
            f.write(f"Final results directory: {final_results_dir}\n")

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
