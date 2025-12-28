#!/usr/bin/env python3
"""
バッチ処理システムのクイックテスト（最初の3枚のみ処理）

Usage:
    nix develop . -c bash -c 'uv run python quick_test_batch.py'
"""

import subprocess
import sys

def main():
    print("🧪 Quick batch test: Processing first 3 BSDS500 images")

    cmd = [
        "uv", "run", "python", "batch_process_bsds500.py",
        "--limit", "3",
        "--output", "results/quick_test_3images"
    ]

    try:
        # Nix環境内で実行
        nix_cmd = ["nix", "develop", ".", "-c", "bash", "-c", " ".join(cmd)]
        result = subprocess.run(nix_cmd, timeout=1800)  # 30分タイムアウト

        if result.returncode == 0:
            print("✅ Quick test completed successfully!")
            print("Check results in: results/quick_test_3images/")
        else:
            print(f"❌ Quick test failed with return code: {result.returncode}")

    except subprocess.TimeoutExpired:
        print("⏰ Quick test timeout")
    except Exception as e:
        print(f"💥 Error during quick test: {e}")

if __name__ == "__main__":
    main()