#!/usr/bin/python
import cv2


# Returns contours of the hand
def getHandContours(frame):
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
    # TODO: Add BLOB analysis for better extraction
    return handContour
