
#from huggingface_hub import from_pretrained_keras
from collections import defaultdict
import glob
import os
import time
from typing import Dict, List, Set
import onnxruntime as rt
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda 

#model = from_pretrained_keras("SmilingWolf/wd-v1-4-convnext-tagger-v2")
import numpy as np

from PIL import Image

import cv2
import numpy as np
from PIL import Image

def detect_gpu_change():
    # Check if the GPU information file exists
    if not os.path.exists('gpu_info.txt'):
        with open('gpu_info.txt', 'w') as f:
            # Write the current GPU information to the file
            f.write(str(cuda.Device(0).name()) + '\n')
            f.write(str(cuda.Device(0).compute_capability()) + '\n')
            f.write(str(cuda.Device(0).total_memory()) + '\n')
        return True

    # Read the stored GPU information from the file
    with open('gpu_info.txt', 'r') as f:
        stored_gpu_name = f.readline().strip()
        stored_gpu_capability = f.readline().strip()
        stored_gpu_memory = int(f.readline().strip())

    # Check the current GPU information
    current_gpu_name = str(cuda.Device(0).name())
    current_gpu_capability = str(cuda.Device(0).compute_capability())
    current_gpu_memory = int(cuda.Device(0).total_memory())

    # Compare the stored and current GPU information
    if (current_gpu_name != stored_gpu_name or
            current_gpu_capability != stored_gpu_capability or
            current_gpu_memory != stored_gpu_memory):
        # Update the GPU information file
        with open('gpu_info.txt', 'w') as f:
            f.write(current_gpu_name + '\n')
            f.write(current_gpu_capability + '\n')
            f.write(str(current_gpu_memory) + '\n')
        return True
    else:
        return False

def smart_imread(img, flag=cv2.IMREAD_UNCHANGED):
    if img.endswith(".gif"):
        img = Image.open(img)
        img = img.convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(img, flag)
    return img


def smart_24bit(img):
    if img.dtype is np.dtype(np.uint16):
        img = (img / 257).astype(np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        trans_mask = img[:, :, 3] == 0
        img[trans_mask] = [255, 255, 255, 255]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img

import pandas as pd
from utils import download_model_file

def download_models() :
    os.makedirs('models', exist_ok = True)
    download_model_file('models/convnext.onnx', 'https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2/resolve/main/model.onnx', '71f06ecb7b9df81d8f271da4d43997ea2ed363cdac29aa64fcb256c9631e656a')
    download_model_file('models/convnextv2.onnx', 'https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2/resolve/main/model.onnx', 'e91daa19cd9e8725125b7d70702d1560855fb687f8d8c4218eddaa821f41834a')
    download_model_file('models/vit.onnx', 'https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2/resolve/main/model.onnx', '8a21cadd1f88a095094cafbffe3028c3cc3d97f4d58c54344c5994bcf48e24ac')
    download_model_file('models/swinv2.onnx', 'https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2/resolve/main/model.onnx', '67740df7ede9a53e50d6e29c6a5c0d6c862f1876c22545d810515bad3ae17bb1')
    download_model_file('models/moat.onnx', 'https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2/resolve/main/model.onnx', 'b8cef913be4c9e8d93f9f903e74271416502ce0b4b04df0ff1e2f00df488aa03')
    download_model_file('models/selected_tags.csv', 'https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2/raw/main/selected_tags.csv', '8c8750600db36233a1b274ac88bd46289e588b338218c2e4c62bbc9f2b516368')

def load_labels() -> list[str]:
    df = pd.read_csv('models/selected_tags.csv')

    tag_names = df["name"].tolist()
    rating_indexes = list(np.where(df["category"] == 9)[0])
    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes

def alloc_buf(engine):
    # host cpu mem
    h_in_size = trt.volume(engine.get_binding_shape(0))
    h_out_size = trt.volume(engine.get_binding_shape(1))
    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
    in_cpu = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    out_cpu = cuda.pagelocked_empty(h_out_size, h_out_dtype)
    # allocate gpu mem
    in_gpu = cuda.mem_alloc(in_cpu.nbytes)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)
    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream

def build_engine(model_filepath: str, logger, bs: int = 1, rebuild: bool = False) :
    basename = os.path.basename(model_filepath)
    dst = os.path.join('engines', basename + '.trt')
    if os.path.exists(dst) and not rebuild :
        return
    print('Building TRT engine for', basename)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(model_filepath)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    assert success
    config = builder.create_builder_config()
    os.makedirs('engines', exist_ok=True)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1 GiB
    serialized_engine = builder.build_serialized_network(network, config)
    with open(dst, 'wb') as f:
        f.write(serialized_engine)

def load_engine(engine_path: str) :
    with open(engine_path, 'rb') as f:
        return f.read()

class MultiTagger :
    def __init__(self) -> None:
        download_models()
        self.tag_names, self.rating_indexes, self.general_indexes, self.character_indexes = load_labels()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        rebuild = detect_gpu_change()
        if rebuild :
            print('GPU change detected, will rebuild TRT engines')
        model_names = ['vit', 'moat', 'convnext', 'convnextv2'] # swinv2 result in Segmentation fault
        for model in model_names :
            build_engine(os.path.join('models', model + '.onnx'), self.logger, rebuild = rebuild)
        self.engines = [self.runtime.deserialize_cuda_engine(load_engine(os.path.join('engines', model + '.onnx.trt'))) for model in model_names]
        self.exe_ctx = [engine.create_execution_context() for engine in self.engines]
        self.memory = [alloc_buf(engine) for engine in self.engines]

    def proprocess_np_bgr(self, image: np.ndarray) :
        height=448

        image = make_square(image, height)
        image = smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)
        return image

    def label(self, path) :
        image = Image.open(path)
        image = image.convert("RGBA")
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, mask=image)
        image = new_image.convert("RGB")
        image = np.asarray(image)

        height=448
        image = image[:, :, ::-1]

        image = make_square(image, height)
        image = smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        probs = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, probs[0].astype(float)))

        general_threshold = 0.35
        general_names = [labels[i] for i in self.general_indexes]
        general_res = [x for x in general_names if x[1] > general_threshold]
        general_res = dict(general_res)

        return general_res
    
    def label_batch(self, img_batch: np.ndarray, threshold = 0.35) :
        per_image_labels: List[List[Dict[str, float]]] = []
        for i in range(img_batch.shape[0]) :
            per_image_labels.append([])
        for img_id, img in enumerate(img_batch) :
            # copy to GPU
            for (exe_ctx, scratch_pad) in zip(self.exe_ctx, self.memory) :
                in_cpu, out_cpu, in_gpu, out_gpu, stream = scratch_pad
                # TODO: 5 HtoD is slow, replace with one HtoD and 4 DtoD in batch
                cuda.memcpy_htod_async(in_gpu, img, stream)
            # run inference
            for (exe_ctx, scratch_pad) in zip(self.exe_ctx, self.memory) :
                in_cpu, out_cpu, in_gpu, out_gpu, stream = scratch_pad
                exe_ctx.execute_async_v2(bindings = [int(in_gpu), int(out_gpu)], stream_handle = stream.handle)
            # copy to CPU
            for (exe_ctx, scratch_pad) in zip(self.exe_ctx, self.memory) :
                in_cpu, out_cpu, in_gpu, out_gpu, stream = scratch_pad
                cuda.memcpy_dtoh_async(out_cpu, out_gpu, stream)
            # wait for complete
            for (exe_ctx, scratch_pad) in zip(self.exe_ctx, self.memory) :
                in_cpu, out_cpu, in_gpu, out_gpu, stream = scratch_pad
                stream.synchronize()
            for (exe_ctx, scratch_pad) in zip(self.exe_ctx, self.memory) :
                in_cpu, out_cpu, in_gpu, out_gpu, stream = scratch_pad
                tag2prob = {tag: prob for tag, prob in zip(self.tag_names, out_cpu)}
                per_image_labels[img_id].append(tag2prob)
        final_tags = []
        for label_sets in per_image_labels :
            merged_dict = {key: max(d[key] for d in label_sets if key in d) for key in set().union(*label_sets)}
            merged_dict = {key: value for key, value in merged_dict.items() if value > threshold}
            final_tags.append(merged_dict)
        return final_tags

TAGGER_WD_ENSEMBLE = None

def tag_image(img_bgr: np.ndarray, threshold: float) :
    global TAGGER_WD_ENSEMBLE
    if TAGGER_WD_ENSEMBLE is None :
        download_models()
        TAGGER_WD_ENSEMBLE = MultiTagger()
    img_in = TAGGER_WD_ENSEMBLE.proprocess_np_bgr(img_bgr)
    return TAGGER_WD_ENSEMBLE.label_batch(img_in)[0]

def test() :
    download_models()

    image = Image.open('1.jpg')
    image = image.convert("RGBA")
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    image = new_image.convert("RGB")
    image = np.asarray(image)

    height=448
    image = image[:, :, ::-1]

    image = make_square(image, height)
    image = smart_resize(image, height)
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)
    #image = np.repeat(image, 16, axis = 0)

    tagger = MultiTagger()
    print(image.shape)
    print(image.min(), image.max())
    perf_measures = []
    for _ in range(10) :
        start = time.perf_counter()
        ret = tagger.label_batch(image)
        perf_measures.append(time.perf_counter() - start)
    print(np.mean(perf_measures))

if __name__ == '__main__' :
    test()
