import cv2
import numpy as np
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
    if check1 % 2 == 0:
        check1 = check1 + 1
    if check2 % 2 == 0:
        check2 = check2 + 1

    median = cv2.GaussianBlur(img, (check1, check2), 0)
    ret,median = cv2.threshold(median,0,255,cv2.THRESH_OTSU)
    return median


# Final callable method
def transform_image(frame):
    frame = tranformToYCC(frame)
    frame = noiseReduction(frame)
    frame = smoothen(frame)
    return frame
