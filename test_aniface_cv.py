import cv2
import sys
import os.path

def detect(filename, cascade_file = "./lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
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
        w = int(w)
        h = int(h)
        x = int(x)
        y = int(y)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # cv2.imshow("AnimeFaceDetect", image)
    # cv2.waitKey(0)
    cv2.imwrite("out.png", image)

detect('dataset/cap_the.magical.revolution.of.the.reincarnated.princess.and.the.genius.young.lady.s01e06.1080p.web.h264-senpai_00_04_33_01.jpg')
