import os
import cv2 as cv
import numpy as np

WIN_UP_X = 0
WIN_UP_Y = 0
WIN_DOWN_X = 300
WIN_DOWN_Y = 300


def video_demo():
    capture = cv.VideoCapture(0)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    while True:
        ret, frame = capture.read()
        ROI = frame[WIN_UP_X:WIN_DOWN_X, WIN_UP_Y:WIN_DOWN_Y]

        ROI = cv.cvtColor(cr_otsu(ROI), cv.COLOR_BGR2GRAY)
        ROI = cv.Canny(ROI, 10, 200)
        ROI = cv.dilate(ROI, kernel)

        cv.imshow('ROI_canny', cv.flip(ROI, 1))
        cv.rectangle(frame, (WIN_UP_X, WIN_UP_Y), (WIN_DOWN_X, WIN_DOWN_Y), (0, 0, 255), 2)
        cv.imshow('video', cv.flip(frame, 1))

        c = cv.waitKey(60)
        if c == 27:
            break


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


def read_gesture(file_path="./Dataset"):
    """
    Read gesture Dataset
    return  X(2062, 100, 100) array , y(2062, ) array
    """
    y = []
    X = np.empty((2062, 100, 100), dtype='float32')
    count = 0
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # 读取手势图片数据集
    for root, dirs, files in os.walk("./Dataset"):
        if len(files):
            for file in files:
                # 读取图片
                img = cv.imread(root + '\\' + file)
                # 设置大小 100 100
                img = cv.resize(img, (100, 100))
                # 肤色识别
                # img = hsv_detect(img)
                img = cv.filter2D(img, -1, kernel=kernel)
                # 转为灰度图片
                img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
                cv.imshow('1', img)
                cv.waitKey(0)

                # 获取数据
                X[count] = np.array(img, dtype='float32')
                count += 1
                y.append(root[-1])

    y = np.array(y)
    return X, y


# read_gesture()
video_demo()
# X, y = read_gesture()   # X(2062, 100, 100)  y(2062, )
# X = X.reshape([-1, 100, 100])
# while True:
#     choice = np.random.choice(2062, 1)
#     img = np.array(X[choice].reshape([100, 100]), dtype=np.uint8)
#     #img = cv.equalizeHist(img)
#     print(y[choice])
#     cv.imshow('test', img)
#     cv.waitKey(0)



