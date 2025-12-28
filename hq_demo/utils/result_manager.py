"""Results directory management and cleanup utilities."""
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple


# デフォルト設定
DEFAULT_MAX_RESULTS = 50  # 保持する最大結果ディレクトリ数
DEFAULT_MAX_AGE_DAYS = 30  # 保持する最大日数


def get_result_directories(results_base: str = "results") -> List[Tuple[Path, datetime]]:
    """
    results/ 内のディレクトリを取得（作成日時付き）

    Args:
        results_base: resultsディレクトリのパス

    Returns:
        [(Path, datetime), ...] のリスト（作成日時でソート）
    """
    results_path = Path(results_base)

    if not results_path.exists():
        return []

    directories = []
    for item in results_path.iterdir():
        if item.is_dir() and item.name != ".git":
            # ディレクトリの作成日時を取得
            mtime = datetime.fromtimestamp(item.stat().st_mtime)
            directories.append((item, mtime))

    # 古い順にソート
    directories.sort(key=lambda x: x[1])

    return directories


def cleanup_old_results(
    results_base: str = "results",
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    max_count: int = DEFAULT_MAX_RESULTS,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    古い結果ディレクトリを自動削除

    削除条件:
    1. max_age_days より古いディレクトリ
    2. max_count を超える古いディレクトリ

    Args:
        results_base: resultsディレクトリのパス
        max_age_days: 保持する最大日数
        max_count: 保持する最大ディレクトリ数
        dry_run: True の場合、実際には削除せずログのみ出力

    Returns:
        (deleted_by_age, deleted_by_count) のタプル
    """
    directories = get_result_directories(results_base)

    if len(directories) == 0:
        print(f"No result directories found in {results_base}")
        return 0, 0

    deleted_by_age = 0
    deleted_by_count = 0

    # 1. 古すぎるディレクトリを削除
    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    remaining_dirs = []

    for dir_path, mtime in directories:
        if mtime < cutoff_date:
            if dry_run:
                print(f"[DRY RUN] Would delete (age): {dir_path.name} ({mtime.strftime('%Y-%m-%d')})")
            else:
                print(f"Deleting old directory: {dir_path.name} ({mtime.strftime('%Y-%m-%d')})")
                shutil.rmtree(dir_path)
            deleted_by_age += 1
        else:
            remaining_dirs.append((dir_path, mtime))

    # 2. 数が多すぎる場合、古いものから削除
    if len(remaining_dirs) > max_count:
        to_delete = remaining_dirs[:len(remaining_dirs) - max_count]

        for dir_path, mtime in to_delete:
            if dry_run:
                print(f"[DRY RUN] Would delete (count): {dir_path.name} ({mtime.strftime('%Y-%m-%d')})")
            else:
                print(f"Deleting excess directory: {dir_path.name} ({mtime.strftime('%Y-%m-%d')})")
                shutil.rmtree(dir_path)
            deleted_by_count += 1

    # サマリー表示
    total_deleted = deleted_by_age + deleted_by_count
    remaining_count = len(directories) - total_deleted

    if dry_run:
        print(f"\n[DRY RUN] Would delete {total_deleted} directories ({deleted_by_age} by age, {deleted_by_count} by count)")
        print(f"[DRY RUN] Would keep {remaining_count} directories")
    else:
        if total_deleted > 0:
            print(f"\n✅ Cleanup complete: deleted {total_deleted} directories ({deleted_by_age} by age, {deleted_by_count} by count)")
            print(f"Remaining: {remaining_count} directories")
        else:
            print(f"No cleanup needed: {remaining_count} directories")

    return deleted_by_age, deleted_by_count


def list_result_directories(results_base: str = "results", verbose: bool = True):
    """
    results/ 内のディレクトリを一覧表示

    Args:
        results_base: resultsディレクトリのパス
        verbose: 詳細情報を表示するか
    """
    directories = get_result_directories(results_base)

    if len(directories) == 0:
        print(f"No result directories found in {results_base}")
        return

    print(f"Found {len(directories)} result directories in {results_base}:\n")

    if verbose:
        for dir_path, mtime in reversed(directories):  # 新しい順に表示
            age_days = (datetime.now() - mtime).days
            print(f"  {dir_path.name:40s} {mtime.strftime('%Y-%m-%d %H:%M')} ({age_days} days old)")
    else:
        for dir_path, _ in reversed(directories):
            print(f"  {dir_path.name}")

    # サイズ情報
    total_size_mb = sum(get_dir_size(d[0]) for d in directories) / (1024 * 1024)
    print(f"\nTotal size: {total_size_mb:.1f} MB")


def get_dir_size(path: Path) -> int:
    """
    ディレクトリのサイズを取得（バイト）

    Args:
        path: ディレクトリパス

    Returns:
        サイズ（バイト）
    """
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except PermissionError:
        pass
    return total
