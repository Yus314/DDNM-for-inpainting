"""Image and tensor conversion utilities."""
import numpy as np
import torch
from PIL import Image
from typing import Tuple


def load_image_as_tensor(
    image_path: str,
    image_size: int = 512,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, Image.Image]:
    """
    画像を読み込んでテンソルに変換

    Args:
        image_path: 画像ファイルのパス
        image_size: リサイズ後のサイズ（正方形）
        device: PyTorchデバイス ('cpu' または 'cuda')

    Returns:
        tuple: (tensor, pil_image)
            - tensor: [-1, 1] に正規化された (1, 3, H, W) テンソル
            - pil_image: PIL形式の画像
    """
    pil_image = Image.open(image_path).convert('RGB')

    if pil_image.size != (image_size, image_size):
        pil_image = pil_image.resize((image_size, image_size), Image.Resampling.LANCZOS)

    # PIL -> numpy -> tensor ([-1, 1] range)
    arr_image = np.array(pil_image).astype(np.float32) / 127.5 - 1
    tensor = torch.from_numpy(np.transpose(arr_image, [2, 0, 1])).unsqueeze(0).to(device)

    return tensor, pil_image


def load_mask_as_tensor(
    mask_path: str,
    mask_size: int = 512,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    マスク画像を読み込んでテンソルに変換

    Args:
        mask_path: マスクファイルのパス
        mask_size: リサイズ後のサイズ（正方形）
        device: PyTorchデバイス

    Returns:
        tuple: (mask_tensor, mask_array)
            - mask_tensor: [0, 1] に正規化された (1, 1, H, W) テンソル
            - mask_array: (H, W, C) の numpy 配列 [0, 1] range
    """
    pil_mask = Image.open(mask_path).convert('RGB')

    if pil_mask.size != (mask_size, mask_size):
        pil_mask = pil_mask.resize((mask_size, mask_size), Image.Resampling.LANCZOS)

    # PIL -> numpy [0, 1] range
    arr_mask = np.array(pil_mask).astype(np.float32) / 255.0

    # numpy -> tensor (use first channel only for mask)
    mask_tensor = torch.from_numpy(arr_mask[:, :, 0:1]).permute(2, 0, 1).unsqueeze(0).to(device)

    return mask_tensor, arr_mask


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    テンソルをPIL画像に変換

    Args:
        tensor: (1, 3, H, W) または (3, H, W) 形式の [-1, 1] 範囲のテンソル

    Returns:
        PIL.Image: RGB画像
    """
    # Handle both (1, 3, H, W) and (3, H, W) formats
    if tensor.dim() == 4:
        arr = tensor[0].cpu().detach().numpy()
    else:
        arr = tensor.cpu().detach().numpy()

    # (C, H, W) -> (H, W, C)
    arr = np.transpose(arr, [1, 2, 0])

    # [-1, 1] -> [0, 255]
    arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def tensor_to_numpy(tensor: torch.Tensor, denormalize: bool = True) -> np.ndarray:
    """
    テンソルをnumpy配列に変換

    Args:
        tensor: (1, 3, H, W) または (3, H, W) 形式のテンソル
        denormalize: True の場合 [-1, 1] -> [0, 255] に変換

    Returns:
        np.ndarray: (H, W, C) 形式の numpy 配列
    """
    # Handle both (1, 3, H, W) and (3, H, W) formats
    if tensor.dim() == 4:
        arr = tensor[0].cpu().detach().numpy()
    else:
        arr = tensor.cpu().detach().numpy()

    # (C, H, W) -> (H, W, C)
    arr = np.transpose(arr, [1, 2, 0])

    if denormalize:
        # [-1, 1] -> [0, 255]
        arr = ((arr + 1) * 127.5).clip(0, 255).astype(np.uint8)

    return arr


def create_comparison_image(
    original: Image.Image,
    result: Image.Image,
    mask: torch.Tensor
) -> Image.Image:
    """
    比較画像を作成（元画像 | マスク | 結果）

    Args:
        original: 元画像
        result: 復元結果画像
        mask: マスクテンソル (1, 1, H, W)

    Returns:
        PIL.Image: 3枚を横に並べた画像
    """
    w, h = original.size

    # マスクをPIL画像に変換
    mask_arr = mask[0, 0].cpu().numpy()
    mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8)).convert('RGB')
    mask_pil = mask_pil.resize((w, h), Image.Resampling.NEAREST)

    # 3枚を横に並べる
    combined = Image.new('RGB', (w * 3, h))
    combined.paste(original, (0, 0))
    combined.paste(mask_pil, (w, 0))
    combined.paste(result, (w * 2, 0))

    return combined
