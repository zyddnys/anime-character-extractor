
import cv2
import numpy as np

CV_DETECTOR = None

def resize_keep_aspect(img: np.ndarray, size: int) :
    ratio = (float(size)/min(img.shape[0], img.shape[1]))
    new_width = round(img.shape[1] * ratio)
    new_height = round(img.shape[0] * ratio)
    return cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR), ratio

def detect(img_rgb_hwc: np.ndarray, detector = 'cv') :
    ret = []
    offset = []
    global CV_DETECTOR, NN_DETECTOR
    if detector == 'cv' :
        if CV_DETECTOR is None :
            CV_DETECTOR = cv2.CascadeClassifier('lbpcascade_animeface.xml')
        img2, ratio = resize_keep_aspect(img_rgb_hwc, 720)
        img2 = img_rgb_hwc
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = CV_DETECTOR.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
        scale = 1.4
        for (x, y, w, h) in faces :
            x = (x + w / 2) - w / scale / 2
            y = (y + h / 2) - h / scale / 2
            w /= scale
            h /= scale
            w = int(w / ratio)
            h = int(h / ratio)
            x = int(x / ratio)
            y = int(y / ratio)
            ret.append(np.array([x, y, x + w, y + h]))
            offset.append(0)
    return np.stack(ret, axis = 0) if ret else [], offset
