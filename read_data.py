import os
import cv2 as cv
import numpy as np

# 数据说明：
# 手势识别的数据集来自于 Turkey Ankara Ayrancı Anadolu High School'sSignLanguageDigitsDataset
# 图像大小：100*100 像素 （部分3024*3024）
# 颜色空间：RGB 种类：
# 图片种类：10 种(0,1,2,3,4,5,6,7,8,9)
# 每种图片数量：约200 张
#
# 共2062张图片


def read_gesture(file_path="./Dataset"):
    """
    Read gesture Dataset
    return  X(2062, 100, 100) array , y(2062, ) array
    """
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    y = []
    X = np.empty((2062, 100, 100), dtype='float32')
    count = 0
    # 读取手势图片数据集
    for root, dirs, files in os.walk("./Dataset"):
        if len(files):
            for file in files:
                # 读取图片
                img = cv.imread(root + '\\' + file)
                # 设置大小 100 100
                img = cv.resize(img, (100, 100))
                # 肤色检测
                img = cv.cvtColor(cr_otsu(img), cv.COLOR_BGR2GRAY)
                img = cv.Canny(img, 10, 200)
                img = cv.dilate(img, kernel)
                # if root[-1] == '5':
                #     cv.imshow('view', img)
                #     cv.waitKey(0)

                X[count] = np.array(img, dtype='float32')
                count += 1
                y.append(root[-1])


    y = np.array(y)
    return X, y


def hsv_detect(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    (_h, _s, _v) = cv.split(hsv)
    skin = np.zeros(_h.shape, dtype=np.uint8)
    (x, y) = _h.shape

    for i in range(0, x):
        for j in range(0, y):
            if (_h[i][j] > 5) and (_h[i][j] < 25) and (_s[i][j] > 30) and (_s[i][j] < 255) and (_v[i][j] > 50) and (
                    _v[i][j] < 255):
                skin[i][j] = 255
            else:
                skin[i][j] = 0

    dst = cv.bitwise_and(img, img, mask=skin)
    # cv.namedWindow("skin", cv.WINDOW_NORMAL)
    # cv.imshow("skin", dst)
    # cv.namedWindow("mask", cv.WINDOW_NORMAL)
    # cv.imshow("mask", skin)
    # cv.namedWindow("original", cv.WINDOW_NORMAL)
    # cv.imshow("original", img)
    return dst


def cr_otsu(img):
    """
    YCrCb颜色空间的Cr分量+Otsu阈值分割
    """
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv.split(ycrcb)
    cr1 = cv.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    dst = cv.bitwise_and(img, img, mask=skin)
    return dst

