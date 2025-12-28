#!/usr/bin/env python3
"""
BSDS500データセット全体への境界リングマスク適用バッチ処理

使用方法:
    python batch_boundary_ring_inpainting.py --input_dir ./data/datasets/gts/BSDS500_168_BLIP_22-44_11
    python batch_boundary_ring_inpainting.py --max_images 5  # テスト用
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from conf_mgt import conf_base
from guided_diffusion import dist_util
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, select_args
from utils import (
    yamlread,
    setup_model,
    load_image_as_tensor,
    tensor_to_pil,
    create_comparison_image,
)
from utils.mask_generator import (
    create_boundary_ring_mask,
    visualize_mask_info,
    save_mask_as_image,
)


# ============================================================================
# Utility functions now imported from utils module
# ============================================================================
# - setup_model: モデルと拡散プロセスのセットアップ
# - load_image_as_tensor: PIL画像をテンソルに変換
# - tensor_to_pil: テンソルをPIL画像に変換
# - create_comparison_image: 比較画像を作成（元画像 | マスク | 結果）
# ============================================================================


def get_image_list(input_dir: str, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')):
    """入力ディレクトリから画像リストを取得"""
    input_path = Path(input_dir)
    images = []
    for ext in extensions:
        images.extend(input_path.glob(f'*{ext}'))
        images.extend(input_path.glob(f'*{ext.upper()}'))
    return sorted(images)


def process_single_image(
    image_path: Path,
    model,
    diffusion,
    gt_keep_mask: torch.Tensor,
    conf,
    device: str,
    output_dir: Path,
    save_comparison: bool = True
) -> dict:
    """単一画像を処理"""
    image_name = image_path.stem
    image_output_dir = output_dir / image_name
    image_output_dir.mkdir(parents=True, exist_ok=True)
    
    result_info = {
        'image_name': image_name,
        'input_path': str(image_path),
        'status': 'pending',
        'start_time': time.time(),
    }
    
    try:
        # 画像読み込み
        gt, pil_original = load_image_as_tensor(str(image_path), image_size=512, device=device)
        
        # 元画像保存
        pil_original.save(image_output_dir / 'original.png')
        
        # Model kwargs
        model_kwargs = {
            'gt': gt,
            'gt_keep_mask': gt_keep_mask.to(device),
            'deg': 'inpainting',
            'save_path': f'temp_batch_{image_name}',
            'y': None,
            'scale': 1,
            'resize_y': False,
            'sigma_y': 0.0,
            'conf': conf
        }
        
        # Inpainting実行
        with torch.no_grad():
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
        
        # 結果保存
        if sample is not None:
            result_pil = tensor_to_pil(sample)
            result_pil.save(image_output_dir / 'result.png')
            
            # 比較画像保存
            if save_comparison:
                comparison = create_comparison_image(pil_original, result_pil, gt_keep_mask)
                comparison.save(image_output_dir / 'comparison.png')
            
            result_info['status'] = 'success'
        else:
            result_info['status'] = 'failed'
            result_info['error'] = 'No sample generated'
            
    except Exception as e:
        result_info['status'] = 'error'
        result_info['error'] = str(e)
        print(f"  ❌ Error processing {image_name}: {e}")
    
    result_info['end_time'] = time.time()
    result_info['duration'] = result_info['end_time'] - result_info['start_time']
    
    return result_info


def create_summary_grid(output_dir: Path, results: list, grid_cols: int = 5):
    """結果のグリッド画像を作成"""
    successful_results = [r for r in results if r['status'] == 'success']
    
    if not successful_results:
        print("No successful results to create grid")
        return
    
    # 最大20枚までグリッド表示
    max_images = min(len(successful_results), 20)
    images_to_show = successful_results[:max_images]
    
    # 画像サイズ取得
    first_result_path = output_dir / images_to_show[0]['image_name'] / 'result.png'
    if not first_result_path.exists():
        return
    
    sample_img = Image.open(first_result_path)
    img_w, img_h = sample_img.size
    
    # グリッドサイズ計算
    grid_rows = (max_images + grid_cols - 1) // grid_cols
    grid_img = Image.new('RGB', (img_w * grid_cols, img_h * grid_rows), (128, 128, 128))
    
    for i, result in enumerate(images_to_show):
        result_path = output_dir / result['image_name'] / 'result.png'
        if result_path.exists():
            img = Image.open(result_path)
            row = i // grid_cols
            col = i % grid_cols
            grid_img.paste(img, (col * img_w, row * img_h))
    
    summary_dir = output_dir / 'summary'
    summary_dir.mkdir(exist_ok=True)
    grid_img.save(summary_dir / 'results_grid.png')
    print(f"📊 Grid summary saved to {summary_dir / 'results_grid.png'}")


def main():
    parser = argparse.ArgumentParser(description='Batch Boundary Ring Inpainting for BSDS500')
    parser.add_argument('--input_dir', type=str, 
                        default='./data/datasets/gts/BSDS500_168_BLIP_22-44_11',
                        help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, 
                        default='results/bsds500_boundary_ring_batch',
                        help='Output directory')
    parser.add_argument('--center_size', type=int, default=168,
                        help='Center region size')
    parser.add_argument('--boundary_width', type=int, default=5,
                        help='Boundary ring width')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (for testing)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip images that already have results')
    parser.add_argument('--no_comparison', action='store_true',
                        help='Do not save comparison images')
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ログファイルパス
    log_path = output_dir / 'summary' / 'processing_log.json'
    log_path.parent.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("🎯 BSDS500 Boundary Ring Inpainting Batch Processing")
    print("=" * 60)
    
    # Step 1: 画像リスト取得
    print(f"\n📁 Input directory: {args.input_dir}")
    image_list = get_image_list(args.input_dir)
    total_images = len(image_list)
    print(f"   Found {total_images} images")
    
    if args.max_images:
        image_list = image_list[:args.max_images]
        print(f"   Processing first {args.max_images} images (--max_images)")
    
    # Skip existing
    if args.skip_existing:
        filtered_list = []
        for img_path in image_list:
            result_path = output_dir / img_path.stem / 'result.png'
            if not result_path.exists():
                filtered_list.append(img_path)
        skipped = len(image_list) - len(filtered_list)
        image_list = filtered_list
        print(f"   Skipping {skipped} existing results (--skip_existing)")
    
    if not image_list:
        print("No images to process!")
        return
    
    print(f"   Will process {len(image_list)} images")
    
    # Step 2: 設定読み込み
    config_path = "confs/inpainting_512_nine_patch_t100_uncond.yml"
    conf = conf_base.Default_Conf()
    conf.update(yamlread(config_path))
    
    print(f"\n📋 Config: {config_path}")
    print(f"   t_T = {conf.schedule_jump_params['t_T']}")
    print(f"   Center size: {args.center_size}x{args.center_size}")
    print(f"   Boundary width: {args.boundary_width}px")
    
    # Step 3: 共通マスク生成
    print(f"\n🎭 Creating boundary ring mask...")
    gt_keep_mask = create_boundary_ring_mask(
        image_size=512,
        center_region_size=args.center_size,
        boundary_width=args.boundary_width,
        device='cpu'
    )
    visualize_mask_info(gt_keep_mask, f"Boundary Ring Mask")
    
    # マスク保存
    masks_dir = output_dir / 'masks'
    masks_dir.mkdir(exist_ok=True)
    save_mask_as_image(gt_keep_mask, str(masks_dir / f'boundary_ring_{args.center_size}_{args.boundary_width}px.png'))
    
    # Step 4: モデルロード
    device = dist_util.dev(conf.get('device'))
    print(f"\n🔧 Loading model...")
    print(f"   Device: {device}")
    
    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print("   Model loaded successfully!")
    
    # Step 5: バッチ処理
    print(f"\n🚀 Starting batch processing...")
    print(f"   Output directory: {output_dir}")
    
    results = []
    start_time = time.time()
    
    for i, image_path in enumerate(tqdm(image_list, desc="Processing images")):
        print(f"\n[{i+1}/{len(image_list)}] Processing: {image_path.name}")
        
        result = process_single_image(
            image_path=image_path,
            model=model,
            diffusion=diffusion,
            gt_keep_mask=gt_keep_mask,
            conf=conf,
            device=device,
            output_dir=output_dir,
            save_comparison=not args.no_comparison
        )
        
        results.append(result)
        
        # 進捗ログ保存（10枚ごと）
        if (i + 1) % 10 == 0:
            with open(log_path, 'w') as f:
                json.dump({
                    'config': {
                        'input_dir': args.input_dir,
                        'output_dir': args.output_dir,
                        'center_size': args.center_size,
                        'boundary_width': args.boundary_width,
                    },
                    'progress': {
                        'processed': i + 1,
                        'total': len(image_list),
                    },
                    'results': results
                }, f, indent=2)
        
        if result['status'] == 'success':
            print(f"  ✅ Completed in {result['duration']:.1f}s")
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    
    total_time = time.time() - start_time
    
    # Step 6: サマリー生成
    print(f"\n📊 Generating summary...")
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] != 'success')
    
    summary = {
        'config': {
            'input_dir': args.input_dir,
            'output_dir': args.output_dir,
            'center_size': args.center_size,
            'boundary_width': args.boundary_width,
            'config_path': config_path,
        },
        'statistics': {
            'total_images': len(image_list),
            'successful': successful,
            'failed': failed,
            'total_time_seconds': total_time,
            'avg_time_per_image': total_time / len(image_list) if image_list else 0,
        },
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    with open(log_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # グリッド画像作成
    create_summary_grid(output_dir, results)
    
    # 完了メッセージ
    print("\n" + "=" * 60)
    print("✅ Batch processing completed!")
    print("=" * 60)
    print(f"   Total images: {len(image_list)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Avg time/image: {total_time/len(image_list):.1f}s")
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"📋 Log saved to: {log_path}")


if __name__ == "__main__":
    main()
