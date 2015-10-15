#!/usr/bin/python
import cv2
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
    cv2.createTrackbar('Ymin', 'YCCCapture', 45, 255, nothing)
    cv2.createTrackbar('Ymax', 'YCCCapture', 96, 255, nothing)
    cv2.createTrackbar('minCr', 'YCCCapture', 115, 255, nothing)
    cv2.createTrackbar('minCb', 'YCCCapture', 110, 255, nothing)
    cv2.createTrackbar('maxCr', 'YCCCapture', 161, 255, nothing)
    cv2.createTrackbar('maxCb', 'YCCCapture', 154, 255, nothing)

    #for denoising
    cv2.createTrackbar('medianValue1', 'Output', 1, 31, nothing)
    cv2.createTrackbar('medianValue2', 'Output', 1, 31, nothing)

    #for kernel
    cv2.createTrackbar('size1', 'Output', 5, 300, nothing)
    cv2.createTrackbar('size2', 'Output', 5, 300, nothing)


# Get configuration values from trackbars for YCC
def getYCCConfig():
    return (
            cv2.getTrackbarPos('Ymin', 'YCCCapture'),
            cv2.getTrackbarPos('Ymax', 'YCCCapture'),
            cv2.getTrackbarPos('minCr', 'YCCCapture'),
            cv2.getTrackbarPos('minCb', 'YCCCapture'),
            cv2.getTrackbarPos('maxCr', 'YCCCapture'),
            cv2.getTrackbarPos('maxCb', 'YCCCapture')
            )


# Return YCC image after transforming input
def tranformToYCC(img):
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    (Ymin,Ymax,minCr,minCb,maxCr,maxCb) = getYCCConfig()
    mincrcb = np.array((Ymin, minCr, minCb))
    maxcrcb = np.array((Ymax, maxCr, maxCb))
    imgYCC = cv2.inRange(imgYCC, mincrcb, maxcrcb)
    return imgYCC


# Cleans the black noise present in the image
def noiseReduction(frame):

    check1 = cv2.getTrackbarPos('size1', 'Output')
    check2 = cv2.getTrackbarPos('size2', 'Output')
    if check1 and check2:
        size1 = check1
        size2 = check2
    else:
        size1 = size2 = 1

    # Used for erorsion and dilation
    kernel = np.ones((size1, size2), np.uint8)
    ### ?
    frame2 = cv2.erode(frame, kernel, iterations = 1)
    frame2 = cv2.dilate(frame, kernel, iterations = 1)
    ### ?
    frame2 = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    frame2 = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    return frame2


# Blurs image to reduce unnecessary contours
def smoothen(img):
    check1 = cv2.getTrackbarPos('medianValue1', 'Output')
    check2 = cv2.getTrackbarPos('medianValue2', 'Output')
    if check1 % 2 == 1 and check2 % 2 == 1:
        value1 = check1
        value2 = check2
    elif check1 % 2 == 1:
        value1 = check1
        value2 = check2 + 1
    elif check2 % 2 == 1:
        value1 = check1 + 1
        value2 = check2
    else:
        value1 = check1 + 1
        value2 = check2 + 1

    median = cv2.GaussianBlur(img, (value1, value2), 0)
    return median


def main():
    # Initialize windows and trackbars
    createWindows()
    addTrackbars()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        frame2 = frame.copy()

        # Get a nice and clean YCC image to work on
        imgYCC = tranformToYCC(frame2)
        imgYCC = noiseReduction(imgYCC)
        imgYCC = smoothen(imgYCC)

        # Marking contours
        newimg = imgYCC.copy()
        contours, hierarchy = cv2.findContours(newimg,cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)

        index, maxArea = 0, 0

        for i in xrange(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > maxArea:
                maxArea = area
                index = i
            realHandContour = contours[index]
            realHandLen = cv2.arcLength(realHandContour, True)
            handContour = cv2.approxPolyDP(realHandContour,
                                    0.001 * realHandLen, True)

        cv2.drawContours(frame2, handContour, -1, (0, 0, 255), -1)
        minX, minY, handWidth, handHeight = cv2.boundingRect(handContour)
        cv2.rectangle(frame2, (minX, minY), (minX + handWidth,
                        minY + handHeight), (0, 255, 0), 2)

        cnt = handContour
        hull = cv2.convexHull(cnt, returnPoints = False)
        defects = cv2.convexityDefects(cnt, hull)
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv2.line(frame2, start, end, [255, 0, 0], 2)
            cv2.circle(frame2, far, 5, [255, 0, 0], -1)

        cv2.drawContours(frame2, handContour, -1, (0, 0, 255), 2)


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
