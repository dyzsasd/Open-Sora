import functools
import json
import logging
import operator
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.datasets.utils import download_url

pretrained_models = {
    "DiT-XL-2-512x512.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt",
    "DiT-XL-2-256x256.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt",
    "Latte-XL-2-256x256-ucf101.pt": "https://huggingface.co/maxin-cn/Latte/resolve/main/ucf101.pt",
    "PixArt-XL-2-256x256.pth": "https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-256x256.pth",
    "PixArt-XL-2-SAM-256x256.pth": "https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-SAM-256x256.pth",
    "PixArt-XL-2-512x512.pth": "https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-512x512.pth",
    "PixArt-XL-2-1024-MS.pth": "https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-1024-MS.pth",
}

def reparameter(ckpt, name=None):
    if "DiT" in name:
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
    elif "Latte" in name:
        ckpt = ckpt["ema"]
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
        del ckpt["temp_embed"]
    elif "PixArt" in name:
        ckpt = ckpt["state_dict"]
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
    return ckpt

def find_model(model_name):
    if model_name in pretrained_models:
        model = download_model(model_name)
        model = reparameter(model, model_name)
        return model
    else:
        assert os.path.isfile(model_name), f"Could not find DiT checkpoint at {model_name}"
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        if "pos_embed_temporal" in checkpoint:
            del checkpoint["pos_embed_temporal"]
        if "pos_embed" in checkpoint:
            del checkpoint["pos_embed"]
        if "ema" in checkpoint:
            checkpoint = checkpoint["ema"]
        return checkpoint

def download_model(model_name):
    assert model_name in pretrained_models
    local_path = f"pretrained_models/{model_name}"
    if not os.path.isfile(local_path):
        os.makedirs("pretrained_models", exist_ok=True)
        web_path = pretrained_models[model_name]
        download_url(web_path, "pretrained_models", model_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model

def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    return logging.getLogger(__name__)

def load_checkpoint(model, ckpt_path, save_as_pt=True):
    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        state_dict = find_model(ckpt_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")
