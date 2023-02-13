
import cv2
import glob
import os
import numpy as np

from animeface_detector import detect as detect_anime_face

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
