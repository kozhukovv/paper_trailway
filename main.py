import os

import pandas as pd
# from pycore.arcspline import Arcspline
from arcspline import Arcspline
import matplotlib.pyplot as plt
import random as rn
import cv2
import numpy as np
from srccam.load_calib import CalibReader
from srccam.calib import Calib
from srccam.camera import Camera
from srccam.point import Point3d as Point
import math
import shutil
# /home/kozhukovv/paper/cv_book/data/asspline24_second02.tsv
FIRST = "./data/asspline24_second02.tsv"
SECOND = "./data/asspline24i_aa.tsv"
# SPLINE_DIR = "./data/trm_depo/splines"
SPLINE_DIR = "./data"
REF = "./data/trm_depo/get.163.182.left/predicted/8.png"
RES_IMG_DIR = "./results"


if __name__ == '__main__':
    
    if not os.path.exists(RES_IMG_DIR):
        os.makedirs(RES_IMG_DIR)
    
    # зачитываем сплайн
    li = []
    for f in os.listdir(SPLINE_DIR):
        if f.endswith('tsv'):
            li.append(pd.read_csv(os.path.join(SPLINE_DIR, f), skiprows=[1], sep='\t'))
    # print(li)
    data = pd.concat(li, axis=0, ignore_index=True)
    # data = pd.read_csv(FIRST, skiprows=[1], sep='\t')
    print(data.shape)
    arc = Arcspline(data)
    arc.build()
    _x, _y = arc.evalute(0.1)
    print(len(_x))
    print(len(_y))

    # зачитываем камеру 
    par = ['K', 'D', 'r', 't']
    # calib_reader = CalibReader('./data/calib_paper/leftImage.yml', param = par)
    calib_reader = CalibReader('./data/calib_paper/leftImage.yml', param = par)
    calib_dict = calib_reader.read()
    calib = Calib(calib_dict)
    camera = Camera(calib)

    # находим случайную точку вокруг
    for index in range(3):
        
        IMAGES_PATH = os.path.join(RES_IMG_DIR, f"dot_{index}")
        if os.path.exists(IMAGES_PATH):
            shutil.rmtree(IMAGES_PATH)
        os.makedirs(IMAGES_PATH)
        # iter = rn.choice(z)
        x_rand = _x[index]
        y_rand = _y[index]

        # выбираем точки вокруг себя
        zone = 50
        fx = (_x < x_rand + zone) & (_x > x_rand - zone)
        fy = (_y < y_rand + zone) & (_y > y_rand - zone)
        f = fx & fy
        x_cam = np.array(_x)[f]
        y_cam = np.array(_y)[f]

        # приводим координаты к исходной точке
        x_vr = x_rand - x_cam
        y_vr = y_rand - y_cam
        xy_vr = np.vstack([x_vr, y_vr])

        # определяем реальную ориентацию
        imsize = [600, 960, 3]

        for alpha in range(361):
            image = np.zeros(imsize)
            # для каждого alpha производим перерасчет координат
            r = np.array([
                [math.cos(alpha), -math.sin(alpha)],
                [math.sin(alpha), math.cos(alpha)]
            ])
            xy_vr_new = r @ xy_vr
            [x_vr_new, y_vr_new] = xy_vr_new
            # избавляемся от отрицательных значений
            f = (x_vr_new > 0) & (y_vr_new > 0)
            x_vr_new = x_vr_new[f]
            y_vr_new = y_vr_new[f]
            count_of_dots = 0
            for (x, y) in zip(x_vr_new, y_vr_new):
                pix = camera.project_point_3d_to_2d(Point((x, y, 0)))
                if pix[0] < 0 and pix[0] >= imsize[1] and pix[1] < 0 and pix[1] >= imsize[0]:
                # if pix[0] < 0 or pix[0] >= imsize[1] or pix[1] < 0 or pix[1] >= imsize[0]:
                        continue
                count_of_dots += 1
                cv2.circle(image, pix, 2, [255, 255, 255], 2, cv2.LINE_AA)
            print(f"Angle: {alpha} \t Count of dots: {count_of_dots}")
            cv2.putText(image, f"alpha = {alpha}", [200, 100], cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 255], 2, cv2.LINE_AA)
            cv2.putText(image, f"Count of dots: {count_of_dots}", [200, 200], cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 255], 2, cv2.LINE_AA)
            # print(image.max())
            # cv2.imshow("Image", image)
            crop_img = image[280-70:, 448-70:]
            print(crop_img.shape)
            cv2.imshow("Image", crop_img)
            # cv2.imshow("Image", cv2.resize(image, (512, 320)))
            

            
            print(IMAGES_PATH + f"/image_{alpha}.png")
            cv2.imwrite(IMAGES_PATH + f"/image_{alpha}.png", image)
            cv2.waitKey(10)
    cv2.destroyAllWindows()
