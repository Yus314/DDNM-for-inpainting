#!/usr/bin/env python3
"""
BSDS500データセット全体に対するスライディング画像補完バッチ処理スクリプト

Usage:
    nix develop . -c bash -c 'uv run python batch_process_bsds500.py'

Features:
- 512x512画像のスライディングウィンドウ補完（256x256ウィンドウ、128pxシフト）
- エラーハンドリングと進行状況表示
- 結果の組織的保存
"""

import os
import glob
import sys
import time
import argparse
from pathlib import Path
import subprocess
from datetime import datetime

def setup_directories(base_output_dir):
    """出力ディレクトリの準備"""
    base_path = Path(base_output_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # ログディレクトリ
    log_dir = base_path / "logs"
    log_dir.mkdir(exist_ok=True)

    return base_path, log_dir

def get_image_list(dataset_dir):
    """BSDS500画像リストの取得"""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # JPGファイルを取得してソート
    image_files = sorted(dataset_path.glob("*.jpg"))
    print(f"Found {len(image_files)} images in {dataset_dir}")

    return image_files

def process_single_image(image_path, output_base, config_path="confs/inet256.yml"):
    """単一画像のスライディング補完処理"""
    image_name = image_path.stem
    output_dir = f"{output_base}/{image_name}_sliding_sr"

    cmd = [
        "uv", "run", "python", "main.py",
        "--config", config_path,
        "--path_y", str(image_path),
        "--save_path", output_dir,
        "--deg", "sr_averagepooling",
        "--scale", "2",
        "--class", "950"
    ]

    print(f"Processing: {image_name}")
    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()

    try:
        # Nix環境内で実行
        nix_cmd = ["nix", "develop", ".", "-c", "bash", "-c", " ".join(cmd)]
        result = subprocess.run(nix_cmd,
                              capture_output=True,
                              text=True,
                              timeout=1800)  # 30分タイムアウト

        end_time = time.time()
        duration = end_time - start_time

        if result.returncode == 0:
            print(f"✅ Success: {image_name} ({duration:.1f}s)")
            return True, duration, None
        else:
            error_msg = f"Process failed with return code {result.returncode}"
            print(f"❌ Failed: {image_name} - {error_msg}")
            print(f"STDERR: {result.stderr[:500]}")
            return False, duration, error_msg

    except subprocess.TimeoutExpired:
        error_msg = "Process timeout (30 minutes)"
        print(f"⏰ Timeout: {image_name} - {error_msg}")
        return False, time.time() - start_time, error_msg

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"💥 Error: {image_name} - {error_msg}")
        return False, time.time() - start_time, error_msg

def write_log(log_file, message):
    """ログファイルに記録"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def main():
    parser = argparse.ArgumentParser(description="BSDS500 Batch Processing")
    parser.add_argument("--dataset", default="data/datasets/gts/BSDS500_168_BLIP_22-44_11",
                       help="Dataset directory path")
    parser.add_argument("--output", default="results/bsds500_batch_sliding",
                       help="Output base directory")
    parser.add_argument("--config", default="confs/inet256.yml",
                       help="Configuration file path")
    parser.add_argument("--start", type=int, default=0,
                       help="Start index (for resuming)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Maximum number of images to process")

    args = parser.parse_args()

    # セットアップ
    try:
        output_base, log_dir = setup_directories(args.output)
        image_files = get_image_list(args.dataset)

        # 処理範囲の決定
        if args.limit:
            end_idx = min(args.start + args.limit, len(image_files))
        else:
            end_idx = len(image_files)

        process_images = image_files[args.start:end_idx]

        # ログファイル
        log_file = log_dir / f"batch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        print(f"\n🚀 Starting batch processing:")
        print(f"   Dataset: {args.dataset}")
        print(f"   Output: {args.output}")
        print(f"   Images: {len(process_images)} ({args.start} to {end_idx-1})")
        print(f"   Config: {args.config}")
        print(f"   Log: {log_file}\n")

        # バッチ処理実行
        total_start = time.time()
        success_count = 0
        failed_count = 0
        total_processing_time = 0

        write_log(log_file, f"Batch processing started: {len(process_images)} images")

        for i, image_path in enumerate(process_images):
            current_idx = args.start + i
            print(f"\n📸 [{current_idx+1}/{len(image_files)}] {image_path.name}")

            success, duration, error = process_single_image(
                image_path, args.output, args.config
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
            avg_time = elapsed / (i + 1) if i > 0 else duration
            remaining = len(process_images) - i - 1
            eta = remaining * avg_time

            print(f"📊 Progress: {i+1}/{len(process_images)} | "
                  f"Success: {success_count} | Failed: {failed_count} | "
                  f"ETA: {eta/60:.1f}min")

        # 最終結果
        total_elapsed = time.time() - total_start

        print(f"\n🏁 Batch processing completed!")
        print(f"   Total time: {total_elapsed/60:.1f} minutes")
        print(f"   Successful: {success_count}/{len(process_images)}")
        print(f"   Failed: {failed_count}/{len(process_images)}")
        print(f"   Average processing time: {total_processing_time/success_count:.1f}s per image" if success_count > 0 else "")
        print(f"   Results saved to: {args.output}")
        print(f"   Log saved to: {log_file}")

        summary = (f"BATCH COMPLETE: {success_count} success, {failed_count} failed, "
                  f"{total_elapsed/60:.1f} minutes total")
        write_log(log_file, summary)

    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()