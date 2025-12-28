# Utils Module Documentation

このディレクトリには、DDNMインペインティングプロジェクトの共通ユーティリティモジュールが含まれています。

## モジュール概要

### 1. `model_loader.py`
モデルと拡散プロセスのセットアップを管理します。

**主な機能:**
- `setup_model(config_path, device=None)`: 設定ファイルからモデルをロードし、初期化

**使用例:**
```python
from utils import setup_model

model, diffusion, conf, device = setup_model("confs/test_inpainting_256.yml")
```

### 2. `image_processing.py`
画像とテンソル間の変換を提供します。

**主な機能:**
- `load_image_as_tensor(image_path, image_size=512, device='cpu')`: PIL画像をテンソルに変換
- `load_mask_as_tensor(mask_path, mask_size=512, device='cpu')`: マスクをテンソルに変換
- `tensor_to_pil(tensor)`: テンソルをPIL画像に変換
- `tensor_to_numpy(tensor, denormalize=True)`: テンソルをNumPy配列に変換
- `create_comparison_image(original, result, mask)`: 比較画像を作成

**使用例:**
```python
from utils import load_image_as_tensor, tensor_to_pil

# 画像をロード
gt, pil_image = load_image_as_tensor("image.jpg", device=device)

# 処理後、PILに変換
result_image = tensor_to_pil(result_tensor)
result_image.save("output.png")
```

### 3. `batch_utils.py`
バッチ処理に必要なユーティリティ関数を提供します。

**主な機能:**
- `get_mask_path(mask_type)`: マスクタイプからファイルパスを取得
- `setup_directories(base_output_dir)`: 出力ディレクトリを準備
- `get_image_list(input_dir)`: 入力画像のリストを取得
- `write_log(log_file, message)`: ログファイルに記録
- `create_batch_log_file(log_dir)`: バッチログファイルを作成

**マスクタイプ:**
- `center_mask`: 中央マスク（512x512）
- `boundary_ring_5px`: 境界リングマスク（5px幅）
- `boundary_ring_10px`: 境界リングマスク（10px幅）

**使用例:**
```python
from utils import get_mask_path, setup_directories, get_image_list

# マスクパスを取得
mask_path = get_mask_path('center_mask')

# ディレクトリをセットアップ
output_base, log_dir, final_results_dir = setup_directories("results/my_batch")

# 画像リストを取得
images = get_image_list("data/input")
```

### 4. `result_manager.py`
results/ディレクトリの自動クリーンアップを管理します。

**主な機能:**
- `get_result_directories(results_base="results")`: 結果ディレクトリを取得
- `cleanup_old_results(results_base="results", max_age_days=30, max_count=50, dry_run=False)`: 古い結果を削除
- `list_result_directories(results_base="results", verbose=True)`: 結果ディレクトリを一覧表示

**使用例:**
```python
from utils import cleanup_old_results, list_result_directories

# 結果ディレクトリを一覧表示
list_result_directories()

# 30日以上古い、または50個を超える結果を削除
cleanup_old_results(max_age_days=30, max_count=50)

# ドライラン（実際には削除しない）
cleanup_old_results(dry_run=True)
```

### 5. `mask_generator.py`
マスク生成ユーティリティ（既存）

**主な機能:**
- `create_boundary_ring_mask()`: 境界リングマスクを生成
- `visualize_mask_info()`: マスク情報を可視化
- `save_mask_as_image()`: マスクを画像として保存

## インポート方法

すべてのユーティリティ関数は`utils`モジュールから直接インポートできます：

```python
from utils import (
    # Model loading
    setup_model,

    # Image processing
    load_image_as_tensor,
    load_mask_as_tensor,
    tensor_to_pil,
    create_comparison_image,

    # Batch utilities
    get_mask_path,
    setup_directories,
    get_image_list,
    write_log,
    create_batch_log_file,

    # Results management
    cleanup_old_results,
    list_result_directories,
)
```

## テンソル形式について

### 画像テンソル
- 形式: `(1, 3, H, W)` - バッチサイズ1、RGB 3チャンネル
- 値の範囲: `[-1, 1]` - 正規化済み
- デバイス: CPU または CUDA

### マスクテンソル
- 形式: `(1, 1, H, W)` - バッチサイズ1、グレースケール1チャンネル
- 値の範囲: `[0, 1]` - 0=補完領域、1=既知領域
- デバイス: CPU または CUDA

## リファクタリングによる改善

**Before (重複コード):**
- `setup_model()`: 6箇所で重複
- `load_image_as_tensor()`: 4箇所で重複
- `get_mask_path()`: 3箇所で重複

**After (一元化):**
- すべての共通機能を`utils/`モジュールに統合
- バッチスクリプトから171行削除
- コードの保守性が大幅に向上

## 関連ファイル

- `__init__.py`: モジュールのエクスポート定義
- プロジェクトルート: `/home/kaki/DDNM/hq_demo/`
- バッチスクリプト: `batch_*.py`
