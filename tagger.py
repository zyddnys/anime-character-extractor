
from typing import Dict
import numpy as np
from regdeepdanbooru import tag_image as reg_tag_image
from wd_ensemble_tagger import tag_image as wd_tag_image

def tag_image(img_bgr: np.ndarray, threshold: float, model = 'reg') -> Dict[str, float] :
    if model == 'reg' :
        return reg_tag_image(img_bgr, threshold)
    elif model == 'wd' :
        return wd_tag_image(img_bgr, threshold)
