

import torch
import clip
import numpy as np
import faiss
from PIL import Image

class Dedup :
    def __init__(self, thres: float = 0.96) -> None:
        self.model, self.preprocess = clip.load("ViT-B/32", device = "cuda")
        self.thres = thres
        self.faiss_index = faiss.IndexFlatIP(512)

    def find_similar(self, img_bgr_np: np.ndarray) -> bool :
        image = self.preprocess(Image.fromarray(img_bgr_np[:, :, ::-1])).unsqueeze(0).to("cuda")
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim = -1, keepdim = True)
        image_features = image_features.view(1, -1)
        image_features = image_features.cpu().numpy()
        dist, _ = self.faiss_index.search(image_features, 1)
        self.faiss_index.add(image_features)
        if dist[0][0] > self.thres :
            return True
        return False

DEDUP_MODULE = None

def run_dedup(img_bgr_np: np.ndarray, thres = 0.96) -> bool :
    global DEDUP_MODULE
    if DEDUP_MODULE is None :
        DEDUP_MODULE = Dedup(thres)
    return DEDUP_MODULE.find_similar(img_bgr_np)
