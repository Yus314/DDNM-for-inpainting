# Batch Sliding Window Inpainting

512x512画像に対してスライディングウィンドウ方式でインペイント処理を行うバッチ処理スクリプト。

## 概要

`batch_sliding_inpainting.py`は、指定したディレクトリ内の画像に対して自動的にスライディングウィンドウインペイントを実行します。

### 主な機能

- **スライディングウィンドウ処理**: 512x512画像を3x3=9個の256x256ウィンドウに分割して処理
- **プログレッシブマスク更新**: 補完済み領域を自動的に追跡し、次のウィンドウで再利用
- **3種類のマスクタイプ**: center_mask、boundary_ring_5px、boundary_ring_10px
- **バッチ処理**: ディレクトリ内の複数画像を自動処理
- **結果の自動集約**: `final_results/`ディレクトリに全結果を自動収集
- **詳細なログ**: 処理状況とエラーを記録

## 使用方法

### 基本コマンド

```bash
python batch_sliding_inpainting.py \
    --input_dir <画像ディレクトリ> \
    --mask_type <マスクタイプ> \
    --output_dir <出力ディレクトリ>
```

### 例

```bash
# center_maskで200枚の画像を処理
python batch_sliding_inpainting.py \
    --input_dir data/datasets/gts/BSDS500_168_BLIP_22-44_11 \
    --mask_type center_mask \
    --output_dir results/batch_sliding_output

# 最初の3枚だけ処理
python batch_sliding_inpainting.py \
    --input_dir data/datasets/gts/BSDS500_168_BLIP_22-44_11 \
    --mask_type center_mask \
    --output_dir results/test_3images \
    --limit 3

# 10枚目から5枚処理
python batch_sliding_inpainting.py \
    --input_dir data/datasets/gts/BSDS500_168_BLIP_22-44_11 \
    --mask_type boundary_ring_5px \
    --output_dir results/batch_10_to_15 \
    --start 10 \
    --limit 5
```

### オプション

| オプション | 必須 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--input_dir` | ○ | - | 入力画像ディレクトリ |
| `--mask_type` | ○ | - | マスクタイプ (center_mask, boundary_ring_5px, boundary_ring_10px) |
| `--output_dir` | ○ | - | 出力ディレクトリ |
| `--config` | × | confs/inet512_sliding_inpainting.yml | 設定ファイルパス |
| `--start` | × | 0 | 開始インデックス（レジューム用） |
| `--limit` | × | None | 最大処理枚数（None=全画像） |

## 出力構成

```
output_dir/
├── final_results/                    # 全画像の最終結果を集約
│   ├── 100007_result.png
│   ├── 100039_result.png
│   └── ...
├── 100007/                           # 個別画像の詳細結果
│   ├── input.png                     # 元画像
│   ├── masked_input.png              # マスク適用後の画像
│   ├── mask.png                      # マスク画像
│   ├── patch_grid.png                # 9パッチのグリッド表示（NEW!）
│   └── boundary_visualization.png    # パッチ境界線可視化（NEW!）
├── 100039/
│   └── ...
├── logs/
│   └── batch_log_YYYYMMDD_HHMMSS.txt
└── summary.txt

# 詳細なパッチ情報（results/results/output_dir/以下）
results/results/output_dir/100007/
└── patches/
    ├── patch_00/
    │   ├── input.png
    │   ├── mask.png
    │   └── progressive/              # プログレッシブサンプリング結果
    │       ├── step_000.png          # 最終結果
    │       ├── step_005.png
    │       ├── step_010.png
    │       └── ...
    ├── patch_01/
    │   └── ...
    └── ... (patch_00 ~ patch_08)
```

## 処理の詳細

### スライディングウィンドウアルゴリズム

1. **初期化**: 512x512画像を読み込み、マスクを適用
2. **ウィンドウ生成**: 3x3=9個の256x256ウィンドウに分割（128pxずつオーバーラップ）
3. **順次処理**: 各ウィンドウを順番に処理
   - ウィンドウ(0,0): [0:256, 0:256]
   - ウィンドウ(0,1): [0:256, 128:384]
   - ウィンドウ(0,2): [0:256, 256:512]
   - ... (計9個)
4. **プログレッシブマスク更新**:
   - 各ウィンドウ処理後、補完した領域をグローバルマスクに反映
   - 次のウィンドウ処理時、すでに補完済みの領域は保持される
5. **最終結果**: マスクカバレッジが100%に達したら完了

### Nine-Patch処理方式

本スクリプトは**Nine-Patch Inpainting Mode**を使用しています：
1. 512x512画像を9つの256x256パッチに分割（128pxオーバーラップ）
2. 各パッチを独立して処理（DDNM拡散モデル）
3. 重なり領域を考慮してブレンド合成
4. 最終的に512x512の完全な補完画像を生成

## 可視化機能

### 1. Patch Grid (patch_grid.png)
9つのパッチを3x3グリッドで表示：
- 各パッチ（P0-P8）の最終結果を並べて表示
- 赤いグリッド線で区切り
- 各パッチの左上に番号を表示
- パッチ間の品質や特徴の違いを確認可能

### 2. Boundary Visualization (boundary_visualization.png)
最終結果にパッチ境界を重ねて表示：
- 各パッチを異なる色の境界線で囲む
  - P0: 赤、P1: 緑、P2: 青
  - P3: 黄、P4: マゼンタ、P5: シアン
  - P6: オレンジ、P7: 紫、P8: 緑青
- パッチのオーバーラップ領域が視覚的に確認可能
- ブレンド処理の品質確認に有用

### 3. プログレッシブサンプリング結果
各パッチの処理過程を確認可能（`results/results/output_dir/image_name/patches/`）：
- `step_000.png`: 最終結果
- `step_005.png ~ step_095.png`: 拡散過程の中間結果
- 5ステップごとに保存（合計20枚）

## パフォーマンス

- **処理速度**: 約24.6秒/画像 (512x512)
- **GPU使用**: CUDA対応GPU推奨
- **メモリ**: 約4GB VRAM

## トラブルシューティング

### エラー: "The size of tensor a (256) must match the size of tensor b (512)"

**原因**: 古いバージョンの`gaussian_diffusion.py`を使用している可能性があります。

**解決策**: `guided_diffusion/gaussian_diffusion.py:734`のA演算子定義を確認してください：

```python
# 正しい定義（動的マスク参照）
A = lambda z: z*model_kwargs.get('gt_keep_mask')

# 誤った定義（マスクをキャプチャ）
mask = model_kwargs.get('gt_keep_mask')
A = lambda z: z*mask
```

### エラー: "CUDA out of memory"

**解決策**:
- GPUメモリを確保してください
- 他のプログラムを終了してください
- バッチサイズ（`--limit`）を小さくして複数回実行してください

## 技術的な詳細

### 処理方式の理解

**重要**: 本スクリプトは「Nine-Patch Inpainting Mode」を使用しています。これは以下の特徴があります：

1. **独立パッチ処理**: 各256x256パッチを独立して拡散モデルで処理
2. **ブレンド合成**: 処理後、オーバーラップ領域を考慮して9パッチをブレンド
3. **高品質**: 各パッチが独立して最適化されるため、細部まで高品質

これは「Progressive Mask Update方式のSliding Window」とは異なります。

### 設定ファイル

デフォルトで`confs/inpainting_512_nine_patch_t100_uncond.yml`を使用します。

主な設定項目:
- `image_size: 256` - ベースモデルサイズ（256x256 U-Net）
- `name: face256` - モデル名
- `timestep_respacing: '100'` - サンプリングステップ数（100ステップ）
- `nine_patch_mode: true` - Nine-Patchモード有効化（重要！）
- `model_path: ./data/pretrained/256x256_diffusion_uncond.pt` - 無条件拡散モデル

### 修正履歴

#### 2025-12-28
- **追加**: パッチ可視化機能
  - `patch_grid.png`: 9パッチのグリッド表示
  - `boundary_visualization.png`: パッチ境界線の可視化
  - 各パッチのプログレッシブサンプリング結果へのアクセス

#### 2025-12-27
- **修正**: Nine-Patch Inpainting Modeの適用
  - `nine_patch_mode: true`フラグの有効化
  - 設定ファイルを`inpainting_512_nine_patch_t100_uncond.yml`に変更
  - shape parameterを`gt.shape`（512x512）に修正

## ライセンス

このプロジェクトはDDNMプロジェクトの一部です。

## 参考

- DDNM論文: [Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model](https://arxiv.org/abs/2212.00490)
- オリジナルリポジトリ: [wyhuai/DDNM](https://github.com/wyhuai/DDNM)
