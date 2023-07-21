
import torch
import torch.nn as nn
import torch.nn.functional as F

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test a trained classification model."""

import argparse
import sys
import cv2

import numpy as np
import pycls.core.losses as losses
import pycls.core.model_builder as model_builder
import pycls.datasets.loader as loader
import pycls.utils.benchmark as bu
import pycls.utils.checkpoint as cu
import pycls.utils.distributed as du
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.multiprocessing as mpu
import pycls.utils.net as nu
import torch
from pycls.core.config import assert_and_infer_cfg, cfg
from pycls.utils.meters import TestMeter

import einops

from utils import download_model_file, resize_keep_aspect_max
def download_models() :
    download_model_file('models/RegNetY-8.0GF_dds_8gpu.yaml', 'https://github.com/zyddnys/RegDeepDanbooru/raw/main/RegNetY-8.0GF_dds_8gpu.yaml', '89ca2f2e1d344aeee29306afaecdaa07ec32115b9c2d9ed6858f42811fe31093')
    download_model_file('models/regdanbooru_labels.txt', 'https://github.com/zyddnys/RegDeepDanbooru/raw/main/danbooru_labels.txt', '9a0d23462ef6e5659f1cb7bdfb26f1face2a7926e0200af55da6003c8d63560b')
    download_model_file('models/regdeepdanbooru2019.ckpt', 'https://github.com/zyddnys/RegDeepDanbooru/releases/download/v1.0/save_4000000.ckpt', 'd5ff5d41e14ebe34665ffb7851df175e521f27467867765e8d41d049ac453cd4')

def build_model():
    # Load config options
    cfg.merge_from_file('models/RegNetY-8.0GF_dds_8gpu.yaml')
    cfg.merge_from_list([])
    assert_and_infer_cfg()
    cfg.freeze()
    # Setup logging
    lu.setup_logging()

    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    # Build the model (before the loaders to speed up debugging)
    model = model_builder.build_model()

    # Load model weights
    #cu.load_checkpoint('RegNetY-8.0GF_dds_8gpu.pyth', model)

    del model.head

    return model

class RegDeepDanbooru(nn.Module) :
    def __init__(self) :
        super(RegDeepDanbooru, self).__init__()
        self.backbone = build_model()
        num_p = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print( 'Backbone has %d parameters' % num_p )
        self.head_danbooru = nn.Linear(2016, 4096)

    def forward_train_head(self, images) :
        """
        images of shape [N, 3, 512, 512]
        """
        with torch.no_grad() :
            feats = self.backbone(images)
            feats = F.adaptive_avg_pool2d(feats, 1).view(-1, 2016)
        danbooru_logits = self.head_danbooru(feats) # [N, 4096]
        return danbooru_logits

    def forward(self, images) :
        """
        images of shape [N, 3, 512, 512]
        """
        feats = self.backbone(images)
        feats = F.adaptive_avg_pool2d(feats, 1).view(-1, 2016)
        danbooru_logits = self.head_danbooru(feats) # [N, 4096]
        return danbooru_logits


class RegDeepDanbooruModel :
    def __init__(self) -> None:
        self.model = RegDeepDanbooru().cuda()
        self.model.load_state_dict(torch.load("models/regdeepdanbooru2019.ckpt")['model'])
        self.model.eval()
        self.label_map = {}
        with open('models/regdanbooru_labels.txt', 'r') as fp :
            for l in fp :
                l = l.strip()
                if l :
                    idx, tag = l.split(' ')
                    self.label_map[int(idx)] = tag

    def predict(self, img: torch.Tensor, threshold: float) :
        with torch.no_grad() :
            danbooru_logits = self.model(img)
        probs = danbooru_logits.sigmoid().cpu()
        choosen_indices = (probs > threshold).nonzero()
        result = {}
        for i in range(probs.size(0)) :
            prob_single = probs[0].numpy()
            indices_single = choosen_indices[choosen_indices[:, 0] == i][:, 1].numpy()
            tag_prob_map = {self.label_map[idx]: prob_single[idx] for idx in indices_single}
        return tag_prob_map

TAGGER_REGDEEPDANBOORU = None

def tag_image(img_bgr: np.ndarray, threshold: float) :
    global TAGGER_REGDEEPDANBOORU
    if TAGGER_REGDEEPDANBOORU is None :
        download_models()
        TAGGER_REGDEEPDANBOORU = RegDeepDanbooruModel()
    img_bgr = resize_keep_aspect_max(img_bgr, 768)
    img = einops.rearrange(torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).float() / 127.5 - 1.0, 'h w c -> 1 c h w').cuda()
    return TAGGER_REGDEEPDANBOORU.predict(img, threshold)
