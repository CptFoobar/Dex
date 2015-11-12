#!/usr/bin/python
import cv2
from gesture_detector import detectGesture
from hand_detector import detectHand
from hand_extractor import getHandContours
from transform_image import transform_image
import numpy as np
from PIL import Image


# No-op function
def nothing(x):
    pass

# Creates windows.
def createWindows():
    # NOTE:All of these will eventually go away since we will
    # not have a window interface except for settings panel
    cv2.namedWindow('YCCCapture', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)


# Attaches trackbars where required
def addTrackbars():
    #for YCrCb
    cv2.createTrackbar('Ymin', 'YCCCapture', 99, 255, nothing)
    cv2.createTrackbar('Ymax', 'YCCCapture', 215, 255, nothing)
    cv2.createTrackbar('minCr', 'YCCCapture', 135, 255, nothing)
    cv2.createTrackbar('minCb', 'YCCCapture', 120, 255, nothing)
    cv2.createTrackbar('maxCr', 'YCCCapture', 153, 255, nothing)
    cv2.createTrackbar('maxCb', 'YCCCapture', 129, 255, nothing)

    #for denoising
    cv2.createTrackbar('medianValue1', 'Output', 5, 31, nothing)
    cv2.createTrackbar('medianValue2', 'Output', 5, 31, nothing)

    #for kernel
    cv2.createTrackbar('size1', 'Output', 10, 300, nothing)
    cv2.createTrackbar('size2', 'Output', 10, 300, nothing)


def main():
    # Ignore CV2 warnings
    np.seterr(invalid='ignore')

    # Initialize windows and trackbars
    createWindows()
    addTrackbars()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        frame2 = frame.copy()

        # Get a nice and clean YCC image to work on
        # TODO: BG subtraction
        imgYCC = transform_image(frame2)

        # Get contours of the hand
        handContour = getHandContours(imgYCC)
        if handContour is None:
            continue

        fingertips, center, rad, handDimens = detectHand(handContour)

        if fingertips is None or center is None:
            continue

        cv2.circle(frame2, tuple(center), 5, [0, 0, 0], 2)
        cv2.circle(frame2, tuple(center), int(rad), [50, 1, 164], 2)

        for tip in fingertips:
            cv2.circle(frame2, tuple(tip), 4, [18, 123, 251], 2)
            cv2.line(frame2, tuple(tip), tuple(center), [245, 157, 100], 2)


        gesture = detectGesture(fingertips, center, handDimens)
        print gesture

        if frame.any():
            cv2.imshow('YCCCapture',imgYCC)
            cv2.imshow('Output',frame2)
        k = cv2.waitKey(10)
        if k == 27 or k == ord('q'):
            break

    # Cleanup
    cap.release()
    del(cap)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
