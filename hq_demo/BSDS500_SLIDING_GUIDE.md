# 🎯 BSDS500 512x512 スライディング画像補完システム

## 📋 概要

DDNM (Data-Driven Nonlinear denoising Model) とスライディングウィンドウ技術を組み合わせた、512x512高解像度画像のinpainting/超解像システムです。

### ✅ 実装済み機能

- **スライディングウィンドウ処理**: 512x512画像を256x256チャンクで処理
- **50%オーバーラップ**: 128ピクセルシフトによる継ぎ目なし補完
- **バッチ処理**: BSDS500データセット全168枚の自動処理
- **エラーハンドリング**: タイムアウト・例外処理付き
- **進行状況追跡**: リアルタイム進行表示とログ記録

## 🚀 使用方法

### 1. 環境セットアップ
```bash
# Nix開発環境に入る
nix develop .

# 依存関係が自動的にロードされます（PyTorch + CUDA環境）
```

### 2. クイックテスト（推奨）
```bash
# 最初の3枚でテスト実行
nix develop . -c bash -c 'uv run python quick_test_batch.py'
```

### 3. 全体バッチ処理
```bash
# 全168枚を処理（推定時間: 8-12時間）
nix develop . -c bash -c 'uv run python batch_process_bsds500.py'
```

### 4. カスタム実行例
```bash
# 特定範囲を処理（10番目から20枚）
nix develop . -c bash -c 'uv run python batch_process_bsds500.py --start 10 --limit 20'

# 異なる出力ディレクトリ
nix develop . -c bash -c 'uv run python batch_process_bsds500.py --output results/custom_run'
```

## 📁 出力構造

```
results/
└── bsds500_batch_sliding/
    ├── logs/
    │   └── batch_log_20251127_180000.txt
    ├── 250047_sliding_sr/
    │   ├── final/00000.png          # 最終512x512結果
    │   ├── 0_0/, 0_1/, 0_2/        # 各ウィンドウの中間結果
    │   ├── 1_0/, 1_1/, 1_2/
    │   ├── 2_0/, 2_1/, 2_2/
    │   ├── Apy/                    # 前処理結果
    │   └── y/                      # 入力データ
    └── [他の画像の結果...]
```

## 📊 技術仕様

### スライディング処理詳細
- **入力**: 512x512 RGB画像
- **ウィンドウサイズ**: 256x256
- **シフト距離**: 128ピクセル（50%オーバーラップ）
- **総ウィンドウ数**: 3×3 = 9ウィンドウ
- **処理順序**: 左上から右下へ順次処理

### パフォーマンス
- **1画像処理時間**: 約3分（9ウィンドウ × 22秒）
- **メモリ使用量**: 8-12GB VRAM（RTX 4080対応）
- **全データセット**: 168枚 × 3分 = 約8.4時間

### エラーハンドリング
- **タイムアウト**: 30分/画像
- **自動リトライ**: 失敗画像のログ記録
- **レジューム**: `--start`オプションで中断箇所から再開可能

## 🔍 結果確認

### 成功例の確認
```bash
# 最終結果の確認
ls results/bsds500_batch_sliding/*/final/

# ログファイルの確認
tail -f results/bsds500_batch_sliding/logs/batch_log_*.txt
```

### 画質評価
```bash
# 結果画像のサイズ確認
file results/bsds500_batch_sliding/*/final/00000.png

# 個別ウィンドウ結果の確認
ls results/bsds500_batch_sliding/250047_sliding_sr/*/
```

## 🛠️ トラブルシューティング

### よくある問題

1. **CUDA Out of Memory**
   - 解決策: より小さなbatch_sizeに調整、または他のプロセスを停止

2. **処理が遅い**
   - 原因: GPU使用率100%は正常、タイムアウトまで待機

3. **ファイルが見つからない**
   - 確認: `data/datasets/gts/BSDS500_168_BLIP_22-44_11/`ディレクトリの存在

### デバッグ情報
```bash
# GPU状況確認
nvidia-smi

# 処理状況確認
ps aux | grep python

# ディスク容量確認
df -h .
```

## 📈 性能最適化オプション

### 高速処理設定
```bash
# より短いtimestep_respacing（品質 vs 速度のトレードオフ）
# confs/inet256.ymlの'timestep_respacing'を'25'に変更
```

### メモリ最適化
```bash
# FP16使用（メモリ削減）
# confs/inet256.ymlの'use_fp16'をtrueに設定
```

## 🎯 次のステップ

1. **結果分析**: 生成画像の品質評価
2. **パラメータ最適化**: より高品質な設定の探索
3. **512x512専用モデル**: より適切なプリトレインモデルの導入
4. **並列化**: 複数GPU対応の実装

---

*このシステムは学術研究目的で開発されています。商用利用には適切なライセンス確認が必要です。*