#!/usr/bin/env python3

"""
512x512画像の9分割Inpainting システム
Nine-Patch Inpainting: 256x256モデルを使用した512x512画像の確実な補完
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from typing import List, Tuple, Dict

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from conf_mgt import conf_base
from guided_diffusion import dist_util
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, select_args
from utils import yamlread

class NinePatchInpainting:
    """512x512画像の9分割inpaintingシステム"""

    def __init__(self, config_path: str = "confs/test_inpainting_256.yml"):
        """
        Initialize the nine-patch inpainting system

        Args:
            config_path: 256x256 inpainting設定ファイルのパス
        """
        self.config_path = config_path
        self.device = None
        self.model = None
        self.diffusion = None
        self.conf = None

        # 9分割の座標定義
        self.patch_coords = [
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

        self._setup_model()

    def _setup_model(self):
        """256x256 inpaintingモデルのセットアップ"""
        # Load configuration
        self.conf = conf_base.Default_Conf()
        self.conf.update(yamlread(self.config_path))

        # Setup device
        self.device = dist_util.dev(self.conf.get('device'))

        # Create model and diffusion
        self.model, self.diffusion = create_model_and_diffusion(
            **select_args(self.conf, model_and_diffusion_defaults().keys()), conf=self.conf
        )

        # Load pretrained model
        print(f"Loading model from {self.conf.model_path}")
        self.model.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(self.conf.model_path), map_location="cpu")
        )
        self.model.to(self.device)
        if self.conf.use_fp16:
            self.model.convert_to_fp16()
        self.model.eval()

    def extract_patch(self, image: torch.Tensor, mask: torch.Tensor,
                     h_start: int, h_end: int, w_start: int, w_end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract a 256x256 patch from 512x512 image and mask

        Args:
            image: Input image tensor [B, C, 512, 512]
            mask: Input mask tensor [B, C, 512, 512]
            h_start, h_end, w_start, w_end: Patch coordinates

        Returns:
            Tuple of (patch_image, patch_mask) both [B, C, 256, 256]
        """
        patch_img = image[:, :, h_start:h_end, w_start:w_end]
        patch_mask = mask[:, :, h_start:h_end, w_start:w_end]

        # Ensure 256x256 size (handle boundary cases)
        if patch_img.shape[2] != 256 or patch_img.shape[3] != 256:
            patch_img = F.interpolate(patch_img, size=(256, 256), mode='bilinear', align_corners=False)
            patch_mask = F.interpolate(patch_mask, size=(256, 256), mode='bilinear', align_corners=False)

        return patch_img, patch_mask

    def inpaint_patch(self, patch_image: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform inpainting on a single 256x256 patch

        Args:
            patch_image: Patch image tensor [B, C, 256, 256]
            patch_mask: Patch mask tensor [B, C, 256, 256]

        Returns:
            Inpainted patch tensor [B, C, 256, 256]
        """
        # Prepare model kwargs
        class_id = torch.tensor([950], device=self.device)  # Default class (orange)

        model_kwargs = {
            'gt': patch_image,
            'gt_keep_mask': patch_mask,
            'deg': 'inpainting',
            'save_path': 'temp_patch_inpainting',
            'y': class_id,
            'scale': 1,
            'resize_y': False,
            'sigma_y': 0.0,
            'conf': self.conf
        }

        # Generate inpainted result
        with torch.no_grad():
            try:
                sample = None
                for sample_dict in self.diffusion.p_sample_loop_progressive(
                    self.model,
                    (1, 3, 256, 256),
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                    device=self.device,
                    conf=self.conf
                ):
                    sample = sample_dict["sample"]

                return sample if sample is not None else patch_image

            except Exception as e:
                print(f"Patch inpainting error: {e}, using original patch")
                return patch_image

    def blend_overlaps(self, patches: List[torch.Tensor]) -> torch.Tensor:
        """
        Blend overlapping patches into a seamless 512x512 image

        Args:
            patches: List of 9 inpainted patches [B, C, 256, 256]

        Returns:
            Blended 512x512 image [B, C, 512, 512]
        """
        B, C = patches[0].shape[:2]
        result = torch.zeros(B, C, 512, 512, device=self.device)
        weight_map = torch.zeros(B, C, 512, 512, device=self.device)

        for i, patch in enumerate(patches):
            h_start, h_end, w_start, w_end = self.patch_coords[i]

            # Create weight for this patch (center-weighted)
            patch_weight = torch.ones_like(patch)

            # Apply gradual weight reduction near borders for overlapping areas
            if i in [1, 4, 7]:  # 横方向オーバーラップあり
                # Left border fade
                patch_weight[:, :, :, :64] *= torch.linspace(0.3, 1.0, 64).view(1, 1, 1, -1).to(self.device)
                # Right border fade
                patch_weight[:, :, :, 192:] *= torch.linspace(1.0, 0.3, 64).view(1, 1, 1, -1).to(self.device)

            if i in [3, 4, 5]:  # 縦方向オーバーラップあり
                # Top border fade
                patch_weight[:, :, :64, :] *= torch.linspace(0.3, 1.0, 64).view(1, 1, -1, 1).to(self.device)
                # Bottom border fade
                patch_weight[:, :, 192:, :] *= torch.linspace(1.0, 0.3, 64).view(1, 1, -1, 1).to(self.device)

            # Add weighted patch to result
            result[:, :, h_start:h_end, w_start:w_end] += patch * patch_weight
            weight_map[:, :, h_start:h_end, w_start:w_end] += patch_weight

        # Normalize by weight map to handle overlaps
        result = result / (weight_map + 1e-8)

        return result

    def process_image(self, image_path: str, mask_path: str, output_dir: str) -> str:
        """
        Process a 512x512 image using nine-patch inpainting

        Args:
            image_path: Path to input 512x512 image
            mask_path: Path to input 512x512 mask
            output_dir: Directory for output results

        Returns:
            Path to the final inpainted result
        """
        print(f"🎯 Starting Nine-Patch Inpainting: {image_path}")

        # Load and prepare images
        pil_image = Image.open(image_path).convert('RGB')
        if pil_image.size != (512, 512):
            print(f"Resizing image from {pil_image.size} to (512, 512)")
            pil_image = pil_image.resize((512, 512), Image.Resampling.LANCZOS)

        pil_mask = Image.open(mask_path).convert('RGB')
        if pil_mask.size != (512, 512):
            print(f"Resizing mask from {pil_mask.size} to (512, 512)")
            pil_mask = pil_mask.resize((512, 512), Image.Resampling.LANCZOS)

        # Convert to tensors
        arr_image = np.array(pil_image).astype(np.float32) / 127.5 - 1  # [-1, 1]
        arr_mask = np.array(pil_mask).astype(np.float32) / 255.0  # [0, 1]

        gt = torch.from_numpy(np.transpose(arr_image, [2, 0, 1])).unsqueeze(0).to(self.device)
        gt_keep_mask = torch.from_numpy(np.transpose(arr_mask, [2, 0, 1])).unsqueeze(0).to(self.device)

        print(f"📊 Input: GT {gt.shape}, Mask {gt_keep_mask.shape}")
        print(f"📊 Mask range: {gt_keep_mask.min().item():.3f} - {gt_keep_mask.max().item():.3f}")

        # Process each patch
        inpainted_patches = []
        print(f"🔄 Processing 9 patches...")

        for i, (h_start, h_end, w_start, w_end) in enumerate(self.patch_coords):
            print(f"  Patch {i+1}/9: [{h_start}:{h_end}, {w_start}:{w_end}]")

            # Extract patch
            patch_img, patch_mask = self.extract_patch(gt, gt_keep_mask, h_start, h_end, w_start, w_end)

            # Inpaint patch
            inpainted_patch = self.inpaint_patch(patch_img, patch_mask)
            inpainted_patches.append(inpainted_patch)

        # Blend patches
        print(f"🎨 Blending patches...")
        final_result = self.blend_overlaps(inpainted_patches)

        # Save results
        os.makedirs(output_dir, exist_ok=True)

        # Save final result
        result_np = ((final_result[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
        result_np = np.clip(result_np, 0, 255)
        result_image = Image.fromarray(result_np)
        result_path = os.path.join(output_dir, "nine_patch_result.png")
        result_image.save(result_path)

        # Save comparison images
        input_image = Image.fromarray(((arr_image + 1) * 127.5).astype(np.uint8))
        input_image.save(os.path.join(output_dir, "input.png"))

        masked_input_np = ((gt * gt_keep_mask)[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5
        masked_input_np = np.clip(masked_input_np.astype(np.uint8), 0, 255)
        masked_input_image = Image.fromarray(masked_input_np)
        masked_input_image.save(os.path.join(output_dir, "masked_input.png"))

        mask_image = Image.fromarray((arr_mask * 255).astype(np.uint8))
        mask_image.save(os.path.join(output_dir, "mask.png"))

        print(f"✅ Nine-Patch Inpainting completed!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📄 Final result: {result_path}")

        return result_path

def main():
    """九分割inpaintingのテスト実行"""
    # Initialize nine-patch inpainting system
    inpainter = NinePatchInpainting("confs/test_inpainting_256.yml")

    # Process test image
    image_path = "data/datasets/gts/BSDS500_168_BLIP_22-44_11/100007.jpg"
    mask_path = "data/datasets/gt_keep_masks/center_mask_512/center_mask_512x512.png"
    output_dir = "nine_patch_results"

    # Perform nine-patch inpainting
    result_path = inpainter.process_image(image_path, mask_path, output_dir)

    print(f"\n🎯 Nine-Patch Inpainting Test Completed!")
    print(f"📊 Method: 512x512 → 9×256x256 → inpainting → blended 512x512")
    print(f"🎨 Result: {result_path}")

if __name__ == "__main__":
    main()