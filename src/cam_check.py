import numpy as np
import cv2 as cv
from find_squares import find_squares

threshold = 150
cap = cv.VideoCapture(0)


while(True):
    # Capture frame-by-frame
    ret, color_img = cap.read()

    # Convert to grayscale
    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

    # find the outer square <- make sure that the camera only sees a border of black and the white square
    squares = find_squares(gray_img, threshold)

    # if we found a square then get the cropping ranges
    if len(squares) > 0:
        xm = squares[0][:, 0] < (gray_img.shape[1]/2)
        ym = squares[0][:, 1] < (gray_img.shape[0]/2)
        x_r = slice(np.amax(squares[0][xm, 0]) + 5, np.amin(squares[0][np.logical_not(xm), 0]) - 5)
        y_r = slice(np.amax(squares[0][ym, 1]) + 5, np.amin(squares[0][np.logical_not(ym), 1]) - 5)

    else:
        x_r = slice(0, gray_img.shape[1])
        y_r = slice(0, gray_img.shape[0])

    # Display the resulting frame
    cv.drawContours(color_img, squares, -1, (0, 255, 0), 1)
    cv.imshow('squares', color_img)
    ch = cv.waitKey(1)

    # do the inversion
    dnn_img = (255 - gray_img[y_r, x_r])
    cv.imshow('crop', dnn_img)
    ch = cv.waitKey(10)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
