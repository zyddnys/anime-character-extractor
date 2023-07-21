
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
from anime_character_detector import detect_character
from wd_ensemble_tagger import MultiTagger
from object_descriptor_parser import create_objects_from_descriptor
from anime_seg import get_seg_image

def unsharp(image):
    gaussian_3 = cv2.GaussianBlur(image, (3, 3), 2.0)
    return cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)

def character_shot_generator(img: np.ndarray, grounding_dino_prompt) :
    boxes = detect_character(img, grounding_dino_prompt)
    for box in boxes :
        [x1, y1, x2, y2] = list(box)
        yield img[y1: y2, x1: x2], (x1, y1), (x2, y2)

def img2tags(tagger: MultiTagger, img_bgr: np.ndarray, thres: float) -> Dict[str, float] :
    img_in = tagger.proprocess_np_bgr(img_bgr)
    return tagger.label_batch(img_in)[0]

@torch.no_grad()
def main(src: str, dst: str, desc: str, format: str = 'jpg') :
    tagger = MultiTagger()
    with open(desc, 'r') as fp :
        configs, objects, postprocess = create_objects_from_descriptor(fp.read())

    tag_threshold = float(configs.cfg['min_prob'])
    min_edge_size = int(configs.cfg['min_size'])
    grounding_dino_prompt = configs.cfg["grounding_dino_prompt"]
    do_segment = configs.cfg["segment"].lower() == "true" if "segment" in configs.cfg else False
    print('tag_threshold', tag_threshold)
    print('min_edge_size', min_edge_size)
    print('grounding_dino_prompt', grounding_dino_prompt)
    print('do_segment', do_segment)

    report_freq = 10
    last_time = time.time()
    next_milestone = last_time + report_freq
    n_frames = 0
    n_frames_cur_milestone = 0
    print('[Main] Process start')
    for frame, filename, counter in video_frame_generator_transnetv2(src) :
        counter2 = 0
        for char_shot, (x1, y1), (x2, y2) in character_shot_generator(frame, grounding_dino_prompt) :
            if char_shot.shape[1] > min_edge_size and char_shot.shape[0] > min_edge_size :
                if do_segment :
                    char_shot, raw_shot, mask = get_seg_image(char_shot)
                tags = img2tags(tagger, char_shot, tag_threshold)
                obj = objects.match(tags)
                if obj is not None :
                    tags = postprocess.apply(obj, tags)
                    target_root = os.path.join(dst, obj)
                    os.makedirs(target_root, exist_ok = True)
                    target = os.path.join(target_root, f'{filename}_{counter}_{counter2}.{format}')
                    if do_segment :
                        target_raw = os.path.join(target_root, f'{filename}_{counter}_{counter2}_raw.{format}')
                        target_mask = os.path.join(target_root, f'{filename}_{counter}_{counter2}_mask.{format}')
                    target_txt = os.path.join(target_root, f'{filename}_{counter}_{counter2}.txt')
                    tags_str = ','.join([t for t in tags.keys() if not t.startswith('rating:')]).replace('_', ' ')
                    with open(target_txt, 'w') as fp :
                        fp.write(tags_str)
                    cv2.imwrite(target, char_shot)
                    if do_segment :
                        cv2.imwrite(target_raw, raw_shot)
                        cv2.imwrite(target_mask, mask)
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
