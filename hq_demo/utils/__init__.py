# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

import yaml
import os
from PIL import Image


def txtread(path):
    path = os.path.expanduser(path)
    with open(path, 'r') as f:
        return f.read()


def yamlread(path):
    return yaml.safe_load(txtread(path=path))

def imwrite(path=None, img=None):
    Image.fromarray(img).save(path)


# ============================================================================
# New utility modules
# ============================================================================

# Model loading utilities
from .model_loader import setup_model

# Image processing utilities
from .image_processing import (
    load_image_as_tensor,
    load_mask_as_tensor,
    tensor_to_pil,
    tensor_to_numpy,
    create_comparison_image,
)

# Batch processing utilities
from .batch_utils import (
    MASK_PATHS,
    get_mask_path,
    setup_directories,
    get_image_list,
    write_log,
    create_batch_log_file,
)

# Results management utilities
from .result_manager import (
    get_result_directories,
    cleanup_old_results,
    list_result_directories,
)


__all__ = [
    # Legacy functions
    'txtread',
    'yamlread',
    'imwrite',
    # Model loading
    'setup_model',
    # Image processing
    'load_image_as_tensor',
    'load_mask_as_tensor',
    'tensor_to_pil',
    'tensor_to_numpy',
    'create_comparison_image',
    # Batch processing
    'MASK_PATHS',
    'get_mask_path',
    'setup_directories',
    'get_image_list',
    'write_log',
    'create_batch_log_file',
    # Results management
    'get_result_directories',
    'cleanup_old_results',
    'list_result_directories',
]
