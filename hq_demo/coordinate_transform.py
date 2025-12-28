#!/usr/bin/env python3

"""
Coordinate System Transformation for Sliding Window Inpainting
座標系変換ユーティリティ - Global(512x512) ↔ Local(256x256)
"""

import torch

class SlidingWindowCoordinates:
    def __init__(self, global_size=512, window_size=256, shift_size=128):
        self.global_size = global_size
        self.window_size = window_size
        self.shift_size = shift_size

        # Calculate total shifts
        self.shift_h_total = ((global_size - window_size) // shift_size) + 1
        self.shift_w_total = ((global_size - window_size) // shift_size) + 1

    def global_to_local_coords(self, shift_h, shift_w):
        """
        Global(512x512) → Local(256x256) 座標変換

        Args:
            shift_h, shift_w: Current window position

        Returns:
            dict: 座標情報
        """
        # Global coordinates for current window
        global_h_start = shift_h * self.shift_size
        global_h_end = global_h_start + self.window_size
        global_w_start = shift_w * self.shift_size
        global_w_end = global_w_start + self.window_size

        # Handle boundary cases
        global_h_end = min(global_h_end, self.global_size)
        global_w_end = min(global_w_end, self.global_size)

        # Adjust start if window goes beyond boundary
        if global_h_end - global_h_start < self.window_size:
            global_h_start = global_h_end - self.window_size
        if global_w_end - global_w_start < self.window_size:
            global_w_start = global_w_end - self.window_size

        return {
            'global_h_start': global_h_start,
            'global_h_end': global_h_end,
            'global_w_start': global_w_start,
            'global_w_end': global_w_end,
            'window_h_size': global_h_end - global_h_start,
            'window_w_size': global_w_end - global_w_start
        }

    def get_overlap_regions(self, shift_h, shift_w):
        """
        Overlap領域の計算 - mask-shift trickのための境界領域特定

        Returns:
            dict: オーバーラップ情報
        """
        overlap = {}

        # Top overlap (previous h window)
        if shift_h > 0:
            overlap['top'] = {
                'local_h_start': 0,
                'local_h_end': self.shift_size,
                'global_h_start': shift_h * self.shift_size - self.shift_size,
                'global_h_end': shift_h * self.shift_size
            }

        # Left overlap (previous w window)
        if shift_w > 0:
            overlap['left'] = {
                'local_w_start': 0,
                'local_w_end': self.shift_size,
                'global_w_start': shift_w * self.shift_size - self.shift_size,
                'global_w_end': shift_w * self.shift_size
            }

        return overlap

    def extract_overlap_data(self, x_temp, shift_h, shift_w):
        """
        x_tempからオーバーラップ領域のデータを安全に抽出

        Args:
            x_temp: Global result tensor [B, C, 512, 512]
            shift_h, shift_w: Current window position

        Returns:
            dict: 抽出されたオーバーラップデータ
        """
        coords = self.global_to_local_coords(shift_h, shift_w)
        overlaps = self.get_overlap_regions(shift_h, shift_w)

        overlap_data = {}

        # Extract top overlap
        if 'top' in overlaps and shift_h > 0:
            top = overlaps['top']
            # Global座標でx_tempから抽出
            global_h_start = coords['global_h_start']
            global_h_end = global_h_start + self.shift_size
            global_w_start = coords['global_w_start']
            global_w_end = coords['global_w_end']

            # Boundary check
            global_h_start = max(0, global_h_start)
            global_h_end = min(self.global_size, global_h_end)
            global_w_start = max(0, global_w_start)
            global_w_end = min(self.global_size, global_w_end)

            if global_h_end > global_h_start and global_w_end > global_w_start:
                overlap_data['top'] = x_temp[:, :, global_h_start:global_h_end, global_w_start:global_w_end]

        # Extract left overlap
        if 'left' in overlaps and shift_w > 0:
            global_h_start = coords['global_h_start']
            global_h_end = coords['global_h_end']
            global_w_start = coords['global_w_start']
            global_w_end = global_w_start + self.shift_size

            # Boundary check
            global_h_start = max(0, global_h_start)
            global_h_end = min(self.global_size, global_h_end)
            global_w_start = max(0, global_w_start)
            global_w_end = min(self.global_size, global_w_end)

            if global_h_end > global_h_start and global_w_end > global_w_start:
                overlap_data['left'] = x_temp[:, :, global_h_start:global_h_end, global_w_start:global_w_end]

        return overlap_data

    def apply_mask_shift_trick(self, x0_t_hat, x_temp, shift_h, shift_w):
        """
        修正版 mask-shift trick - 座標系整合性を保証

        Args:
            x0_t_hat: Current window prediction [B, C, 256, 256]
            x_temp: Global accumulated result [B, C, 512, 512]
            shift_h, shift_w: Current window position

        Returns:
            torch.Tensor: Modified x0_t_hat with overlap regions
        """
        if shift_h == 0 and shift_w == 0:
            return x0_t_hat  # First window, no overlap

        overlap_data = self.extract_overlap_data(x_temp, shift_h, shift_w)
        result = x0_t_hat.clone()

        # Apply top overlap
        if 'top' in overlap_data:
            top_data = overlap_data['top']
            if top_data.shape[2] == self.shift_size and top_data.shape[3] == self.window_size:
                result[:, :, 0:self.shift_size, :] = top_data

        # Apply left overlap
        if 'left' in overlap_data:
            left_data = overlap_data['left']
            if left_data.shape[2] == self.window_size and left_data.shape[3] == self.shift_size:
                result[:, :, :, 0:self.shift_size] = left_data

        return result

# Test and validation functions
def test_coordinate_transform():
    """座標変換の検証テスト"""
    coords = SlidingWindowCoordinates()

    print("=== 座標変換テスト ===")
    for shift_h in range(3):
        for shift_w in range(3):
            result = coords.global_to_local_coords(shift_h, shift_w)
            print(f"Window({shift_h},{shift_w}): Global[{result['global_h_start']}:{result['global_h_end']}, {result['global_w_start']}:{result['global_w_end']}]")

    print("\n=== オーバーラップテスト ===")
    dummy_x_temp = torch.randn(1, 3, 512, 512)
    dummy_x0_t_hat = torch.randn(1, 3, 256, 256)

    result = coords.apply_mask_shift_trick(dummy_x0_t_hat, dummy_x_temp, 1, 1)
    print(f"mask-shift trick適用結果サイズ: {result.shape}")
    print("テスト完了 - エラーなし")

if __name__ == "__main__":
    test_coordinate_transform()