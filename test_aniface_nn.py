import cv2
import numpy as np
import PIL
from PIL import Image

from anime_face_detector import create_detector

face_score_threshold = 0.8
landmark_score_threshold = 0.8

detector = create_detector('yolov3')
files = [
    'dataset/cap_the.magical.revolution.of.the.reincarnated.princess.and.the.genius.young.lady.s01e06.1080p.web.h264-senpai_00_17_39_34.jpg',
    'dataset/cap_the.magical.revolution.of.the.reincarnated.princess.and.the.genius.young.lady.s01e03.1080p.web.h264-senpai_00_04_34_232.jpg',
    'dataset/cap_the.magical.revolution.of.the.reincarnated.princess.and.the.genius.young.lady.s01e03.1080p.web.h264-senpai_00_04_35_248.jpg',
    'dataset/cap_the.magical.revolution.of.the.reincarnated.princess.and.the.genius.young.lady.s01e06.1080p.web.h264-senpai_00_04_33_01.jpg',
    'dataset/cap_the.magical.revolution.of.the.reincarnated.princess.and.the.genius.young.lady.s01e06.1080p.web.h264-senpai_00_07_47_38.jpg',
    'dataset/cap_the.magical.revolution.of.the.reincarnated.princess.and.the.genius.young.lady.s01e06.1080p.web.h264-senpai_00_09_01_52.jpg',
    'dataset/cap_the.magical.revolution.of.the.reincarnated.princess.and.the.genius.young.lady.s01e03.1080p.web.h264-senpai_00_03_34_122.jpg',
    'dataset/cap_the.magical.revolution.of.the.reincarnated.princess.and.the.genius.young.lady.s01e03.1080p.web.h264-senpai_00_02_53_01.jpg',
]
for x, f in enumerate(files) :
    image = cv2.imread(f)
    res = image.copy()
    preds = detector(image)
    for pred in preds :
        box = pred['bbox']
        box, score = box[:4], box[4]
        if score < face_score_threshold:
            continue
        box = np.round(box).astype(int)

        lt = max(2, int(3 * (box[2:] - box[:2]).max() / 256))

        height = box[3] - box[1]
        width = box[2] - box[0]
        # print('width', width, 'height', height)
        cv2.rectangle(res, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), lt)

        pred_pts = pred['keypoints']
        counter = 0
        pnp_2d_points = []
        points_3D = np.array([
            (0.0, 0.0, 0.0),       #Nose tip
            (0.0, -330.0, -65.0),  #Chin
            (-225.0, 170.0, -135.0),#Left eye corner
            (225.0, 170.0, -135.0), #Right eye corner 
            (-150.0, -150.0, -125.0),#Left mouth 
            (150.0, -150.0, -125.0) #Right mouth 
        ])
        dist_coeffs = np.zeros((4,1))
        focal_length = image.shape[1]
        center = (image.shape[1]/2, image.shape[0]/2)
        camera_matrix = np.array(
                                [[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], dtype = np.float64
                                )
        pnp_2d_points = [
            pred_pts[23][:2],
            pred_pts[2][:2],
            pred_pts[11][:2],
            pred_pts[19][:2],
            pred_pts[24][:2],
            pred_pts[26][:2]
        ]
        face_left_dist = np.sqrt(np.linalg.norm(pred_pts[23][:2] - pred_pts[1][:2]))
        face_right_dist = np.sqrt(np.linalg.norm(pred_pts[23][:2] - pred_pts[3][:2]))
        face_direction_ratio = (face_left_dist - face_right_dist) / height
        print(face_direction_ratio)
        counter = 0
        for *pt, score in pred_pts:
            if score < landmark_score_threshold:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            pt = np.round(pt).astype(int)
            cv2.circle(res, tuple(pt), lt, color, cv2.FILLED)
            cv2.putText(res, f'{counter}', (pt[0] - 5, pt[1] - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,255), 2)
            counter += 1
        pnp_2d_points = np.stack(pnp_2d_points, axis = 0).astype(np.float64)
        success, rotation_vector, translation_vector = cv2.solvePnP(points_3D, pnp_2d_points, camera_matrix, dist_coeffs, flags=0)
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        p1 = ( int(pnp_2d_points[0][0]), int(pnp_2d_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(res, p1, p2, (255,0,0), 2)
        up_ext = 1.5
        down_ext = 6.4
        h_ext = 1.45
        
        x1, y1 = tuple(box[:2])
        x2, y2 = tuple(box[2:])
        w, h = x2 - x1, y2 - y1
        y1 -= up_ext * (h / 2)
        y2 += down_ext * (h / 2)
        x1 -= h_ext * (w / 2)
        x2 += h_ext * (w / 2)
        offset = face_direction_ratio * 2000
        x1 -= offset
        x2 -= offset
        y1 = int(max(0, y1))
        y2 = int(min(res.shape[0] - 1, y2))
        x1 = int(max(0, x1))
        x2 = int(min(res.shape[1] - 1, x2))
        cv2.rectangle(res, (x1, y1), (x2, y2), (0, 255, 0), 2)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    PIL.Image.fromarray(res).save(f'nn-{x}.png')
