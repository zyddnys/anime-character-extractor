
import cv2
import glob
import os
import numpy as np
import torch
import ffmpeg

from animeface_detector import detect as detect_anime_face
from transnetv2_pytorch import TransNetV2

def resize_keep_aspect_and_pad(img: np.ndarray, size: int, dim: int):
    ratio = size / min(img.shape[0], img.shape[1])
    new_width = round(img.shape[1] * ratio)
    new_height = round(img.shape[0] * ratio)
    img2 = cv2.resize(img, (new_width, new_height), cv2.INTER_LANCZOS4)
    if img2.shape[1] % dim != 0 :
        img2 = cv2.resize(img2, (img2.shape[1] - (img2.shape[1] % dim), img2.shape[0]), cv2.INTER_LANCZOS4)
    elif img2.shape[0] % dim != 0 :
        img2 = cv2.resize(img2, (img2.shape[1], img2.shape[0] - (img2.shape[0] % dim)), cv2.INTER_LANCZOS4)
    return img2

def resize_keep_aspect(img: np.ndarray, size: int):
    ratio = size / min(img.shape[0], img.shape[1])
    new_width = round(img.shape[1] * ratio)
    new_height = round(img.shape[0] * ratio)
    img2 = cv2.resize(img, (new_width, new_height), cv2.INTER_LANCZOS4)
    return img2

def video_frame_generator(path: str, verbose = False) :
    if os.path.isdir(path) :
        files = glob.glob(os.path.join(path, '*.*'))
    else :
        files = [path]
    for f in files :
        (_, filename) = os.path.split(f)
        try :
            vid = cv2.VideoCapture(f)
        except Exception :
            continue
        ctr = 0
        while True :
            # Capture the video frame
            # by frame
            ret, frame = vid.read()
            if not ret :
                break
            frame = resize_keep_aspect(frame, 640)
        
            # Display the resulting frame
            coords, _ = detect_anime_face(frame, detector = 'cv')
            if verbose :
                display_image = frame.copy()
                for coord in coords :
                    x1, y1 = tuple(coord[:2])
                    x2, y2 = tuple(coord[2:])
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.imshow('frame', display_image)
            
                # the 'q' button is set as the
                # quitting button you may use any
                # desired button of your choice
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if len(coords) > 0 and ctr % 12 == 0 :
                yield frame, filename, ctr
            ctr += 1
        vid.release()

def image_files_generator(path: str) :
    files = glob.glob(os.path.join(path, '*.jpg'))
    for f in files :
        print('Processing', f)
        ctr = 0
        (_, filename) = os.path.split(f)
        yield cv2.imread(f), filename, ctr
        ctr += 1

TRANSNETV2 = None


def _predict_raw(model, frames: np.ndarray):
    # [B, T, H, W, 3] to [B, T, 3, H, W]
    single_frame_pred, _ = model(torch.from_numpy(frames).cuda())
    
    single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()

    return single_frame_pred

def _predict_frames(model, frames: np.ndarray):
    def input_iterator():
        # return windows of size 100 where the first/last 25 frames are from the previous/next batch
        # the first and last window must be padded by copies of the first and last frame of the video
        no_padded_frames_start = 25
        no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

        start_frame = np.expand_dims(frames[0], 0)
        end_frame = np.expand_dims(frames[-1], 0)
        padded_inputs = np.concatenate(
            [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
        )

        ptr = 0
        while ptr + 100 <= len(padded_inputs):
            out = padded_inputs[ptr:ptr + 100]
            ptr += 50
            yield out[np.newaxis]

    predictions = []

    for inp in input_iterator():
        single_frame_pred = _predict_raw(model, inp)
        predictions.append(single_frame_pred[0, 25:75, 0],
                            )

        print("\r[TransNetV2] Processing video frames {}/{}".format(
            min(len(predictions) * 50, len(frames)), len(frames)
        ), end="")
    print("")

    single_frame_pred = np.concatenate([single_ for single_ in predictions])

    return single_frame_pred[:len(frames)]

def _video_shot_generator(model, video_filename: str, frame_size: int = 720, max_shot_length_sec: int = 120) :
    probe = ffmpeg.probe(video_filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    fps = float(eval(video_info['r_frame_rate']))
    fps = int(np.ceil(fps))
    width = int(video_info['width'])
    height = int(video_info['height'])
    ratio = frame_size / min(width, height)
    new_width = round(width * ratio)
    new_height = round(height * ratio)
    shot_buffer = np.zeros((max_shot_length_sec * fps, new_height, new_width, 3), dtype = np.uint8)
    video_stream, err = ffmpeg.input(video_filename).output(
        "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
    ).run(capture_stdout=True, capture_stderr=True)
    video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
    single_frame_pred = _predict_frames(model, video)
    vid = cv2.VideoCapture(video_filename)
    ctr = -1
    shot_frame_counter = 0
    while True :
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        if not ret :
            break
        ctr += 1
        frame = resize_keep_aspect(frame, frame_size)
        shot_trans_prob = single_frame_pred[ctr]
        shot_buffer[shot_frame_counter] = frame
        shot_frame_counter += 1
        if shot_frame_counter >= max_shot_length_sec * fps or shot_trans_prob > 0.5 :
            if shot_frame_counter > 0 :
                yield shot_buffer[: shot_frame_counter], ctr - shot_frame_counter + 1
            shot_frame_counter = 0
    if shot_frame_counter > 0 :
        yield shot_buffer[: shot_frame_counter], ctr - shot_frame_counter + 1

def video_frame_generator_transnetv2(path: str, verbose = False) :
    global TRANSNETV2
    TRANSNETV2 = TransNetV2()
    TRANSNETV2.load_state_dict(torch.load("transnetv2-pytorch-weights.pth"))
    TRANSNETV2 = TRANSNETV2.eval().cuda()
    if os.path.isdir(path) :
        files = glob.glob(os.path.join(path, '*.*'))
    else :
        files = [path]
    for f in files :
        try :
            (_, filename) = os.path.split(f)
            for shot_frames, shot_start_frmae_index in _video_shot_generator(TRANSNETV2, f) :
                selected_shot_local_index = 0
                mid_idx = len(shot_frames) // 2
                found = False
                for i in range(mid_idx, len(shot_frames)) :
                    coords, _ = detect_anime_face(shot_frames[i], detector = 'cv')
                    if len(coords) > 0 :
                        selected_shot_local_index = i
                        found = True
                        break
                if found :
                    yield shot_frames[selected_shot_local_index], filename, shot_start_frmae_index + selected_shot_local_index
        except Exception :
            import traceback
            traceback.print_exc()

def test() :
    probe = ffmpeg.probe("dataset\\the.magical.revolution.of.the.reincarnated.princess.and.the.genius.young.lady.s01e01.1080p.web.h264-senpai.mkv")
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    fps = float(eval(video_info['r_frame_rate']))
    print(fps)
    print(video_info)

if __name__ == '__main__' :
    test()
