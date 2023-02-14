
import os
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import einops
import argparse

from video_reader import image_files_generator, video_frame_generator, resize_keep_aspect_and_pad, video_frame_generator_transnetv2
from animeface_detector import detect as detect_face
from deep_danbooru_model import RegDeepDanbooruModel, resize_keep_aspect_max
from object_descriptor_parser import create_objects_from_descriptor

def unsharp(image):
    gaussian_3 = cv2.GaussianBlur(image, (3, 3), 2.0)
    return cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)

def character_shot_generator(img: np.ndarray) :
    coords_nn, offsets = detect_face(img, 'nn')
    up_ext = 1.5
    down_ext = 11.4
    h_ext = 1.45
    for coord, offset in zip(coords_nn, offsets) :
        x1, y1 = tuple(coord[:2])
        x2, y2 = tuple(coord[2:])
        w, h = x2 - x1, y2 - y1
        y1 -= up_ext * (h / 2)
        y2 += down_ext * (h / 2)
        x1 -= h_ext * (w / 2)
        x2 += h_ext * (w / 2)
        x1 -= offset
        x2 -= offset
        y1 = int(max(0, y1))
        y2 = int(min(img.shape[0] - 1, y2))
        x1 = int(max(0, x1))
        x2 = int(min(img.shape[1] - 1, x2))
        yield img[y1: y2, x1: x2], (x1, y1), (x2, y2)

def img2tags(model: RegDeepDanbooruModel, img_bgr: np.ndarray) -> Dict[str, float] :
    img_bgr = resize_keep_aspect_max(img_bgr, 768)
    img = einops.rearrange(torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).float() / 127.5 - 1.0, 'h w c -> 1 c h w').cuda()
    return model.predict(img)

@torch.no_grad()
def main(src: str, dst: str, desc: str, format: str = 'jpg') :
    deepbooru_model = RegDeepDanbooruModel()
    with open(desc, 'r') as fp :
        objects, postprocess = create_objects_from_descriptor(fp.read())
    report_freq = 10
    last_time = time.time()
    next_milestone = last_time + report_freq
    n_frames = 0
    n_frames_cur_milestone = 0
    for frame, filename, counter in video_frame_generator_transnetv2(src) :
        counter2 = 0
        display_image = frame.copy()
        for char_shot, (x1, y1), (x2, y2) in character_shot_generator(frame) :
            if char_shot.shape[1] > 30 and char_shot.shape[0] > 30 :
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.imshow('frame', display_image)
                cv2.waitKey(1)
                tags = img2tags(deepbooru_model, char_shot)
                print(tags)
                obj = objects.match(tags)
                if obj is not None :
                    tags = postprocess.apply(obj, tags)
                    target_root = os.path.join(dst, obj)
                    os.makedirs(target_root, exist_ok = True)
                    target = os.path.join(target_root, f'{filename}_{counter}_{counter2}.{format}')
                    target_txt = os.path.join(target_root, f'{filename}_{counter}_{counter2}.txt')
                    tags_str = ','.join([t for t in tags.keys() if not t.startswith('rating:')]).replace('_', ' ')
                    with open(target_txt, 'w') as fp :
                        fp.write(tags_str)
                    cv2.imwrite(target, char_shot)
                    counter2 += 1
        n_frames += 1
        n_frames_cur_milestone += 1
        cur_time = time.time()
        if cur_time > next_milestone :
            elapsed = cur_time - last_time
            next_milestone = cur_time + report_freq
            print(f'Total Frames={n_frames}, fps={n_frames_cur_milestone / elapsed}')
            n_frames_cur_milestone = 0

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = 'Extract anime characters from anime')
    parser.add_argument('-i', '--input', default='', type=str, help='Path to folder containing list of anime video files')
    parser.add_argument('-d', '--descriptor', default='', type=str, help='Path to an object descriptor file describing which set of anime characters to extract')
    parser.add_argument('-o', '--output', default='', type=str, help='Path to a folder to which extract anime characters are saved')
    args = parser.parse_args()

    main(args.input, args.output, args.descriptor)
