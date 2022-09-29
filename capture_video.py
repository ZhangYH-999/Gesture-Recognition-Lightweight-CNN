import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
from read_data import cr_otsu

WIN_UP_X = 0
WIN_UP_Y = 0
WIN_DOWN_X = 300
WIN_DOWN_Y = 300

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))


def video_demo():
    capture = cv.VideoCapture(0)
    model = load_model("gesture_model.h5")
    cv.namedWindow("App", cv.WINDOW_NORMAL)

    cv.resizeWindow("App", 600, 600)

    while True:
        ret, frame = capture.read()
        # frame = cv.flip(frame, 1)
        # # scr = hsv_detect(frame)
        # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # lower_hsv = np.array([7, 28, 50])
        # upper_hsv = np.array([20, 255, 255])
        # mask = cv.inRange(hsv, lower_hsv, upper_hsv)
        # scr = cv.bitwise_and(frame, frame, mask=mask)
        # 截取手势窗口
        ROI = frame[WIN_UP_X:WIN_DOWN_X, WIN_UP_Y:WIN_DOWN_Y]

        # 转灰度图像
        ROI = cr_otsu(ROI)
        ROI = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)
        ROI = cv.Canny(ROI, 10, 200)
        ROI = cv.dilate(ROI, kernel)
        # 显示
        cv.imshow('ROI', cv.flip(ROI, 1))

        # 预测
        ROI = cv.resize(ROI, (100, 100))
        test_img = np.array(ROI).reshape(1, 100, 100, 1)
        y = model.predict(test_img/255)

        cv.rectangle(frame, (WIN_UP_X, WIN_UP_Y), (WIN_DOWN_X, WIN_DOWN_Y), (0, 255, 0), 3)
        frame = cv.flip(frame, 1)
        cv.putText(frame, "predict is: " + str(np.argmax(y)), (0, 400), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv.imshow('App', frame)
        c = cv.waitKey(60)
        if c == 27:
            break
    cv.destroyAllWindows()


if __name__ == '__main__':

    video_demo()


