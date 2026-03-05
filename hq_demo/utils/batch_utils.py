"""Batch processing utilities."""
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple


# ============================================================================
# マスク管理システム
# ============================================================================

MASK_PATHS = {
    'center_mask': 'data/datasets/gt_keep_masks/center_mask_512/center_mask_512x512.png',
    'boundary_ring_5px': 'data/datasets/gt_keep_masks/boundary_ring_168_5px/mask.png',
    'boundary_ring_10px': 'data/datasets/gt_keep_masks/boundary_ring_168_10px/mask.png',
    'boundary_ring_136_10px': 'data/datasets/gt_keep_masks/boundary_ring_136_10px/mask.png',
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
    image_files = sorted(image_files, reverse=True)

    if len(image_files) == 0:
        raise ValueError(f"No image files found in {input_dir}")

    print(f"Found {len(image_files)} images in {input_dir}")

    return image_files


# ============================================================================
# ログ管理
# ============================================================================

def write_log(log_file: Path, message: str):
    """
    ログファイルに記録

    Args:
        log_file: ログファイルのパス
        message: ログメッセージ
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


def create_batch_log_file(log_dir: Path) -> Path:
    """
    バッチ処理用のログファイルを作成

    Args:
        log_dir: ログディレクトリ

    Returns:
        作成されたログファイルのパス
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_log_{timestamp}.txt"
    return log_file
