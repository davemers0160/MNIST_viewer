import cv2 as cv
import numpy as np


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


def find_squares(img, threshold):
    squares = []
    # img = cv.GaussianBlur(img, (5, 5), 0)

    _, bin_img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)

    bin_img = cv.Canny(bin_img, 0, 50, apertureSize=5, L2gradient=False)
    bin_img = cv.dilate(bin_img, None)

    # cv.imshow('frame', bin)
    # cv.waitKey(1)

    contours, _hierarchy = cv.findContours(bin_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cnt_len = cv.arcLength(cnt, True)
        cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)

        if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4]) for i in range(4)])

            if max_cos < 0.1:
                squares.append(cnt)

    return squares
