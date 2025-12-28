"""Model loading and setup utilities."""
import torch
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    select_args
)
from conf_mgt import conf_base
from utils import yamlread


def setup_model(config_path, device=None):
    """
    モデルと拡散プロセスをセットアップ

    Args:
        config_path: 設定ファイルのパス (YAML)
        device: 使用するデバイス（None の場合は自動検出）

    Returns:
        tuple: (model, diffusion, conf, device)
    """
    print(f"Loading configuration from {config_path}")
    conf_arg = conf_base.Default_Conf()
    conf_arg.update(yamlread(config_path))
    conf = conf_arg

    if device is None:
        device = dist_util.dev(conf.get('device'))

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()),
        conf=conf
    )

    print(f"Loading model weights from {conf.model_path}")
    model.load_state_dict(
        dist_util.load_state_dict(conf.model_path, map_location="cpu")
    )

    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    print(f"✅ Model loaded successfully on {device}")

    return model, diffusion, conf, device
