#!/usr/bin/python
import cv2
from gesture_detector import detectGesture
from hand_detector import detectHand
from hand_extractor import getHandContours
from transform_image import transform_image
from backSubtract import BackgroundSubtractorSuperMOG,applyChoice
import numpy as np
from PythonServer import *

detected_gesture = ""
#Get max Gesture
def maxGestureDetected(n):
    if n == 0:
        return "open-hand"
    elif n == 1:
        return "closed-hand"
    elif n == 2:
        return "fist"
    elif n == 3:
        return "pointing_x"
    elif n == 4:
        return "pointing-y"
    elif n == 5:
        return "metal"
    elif n == 6:
        return "gun"
    elif n == 7:
        return "two"
    elif n == 8:
        return "three"
    elif n == 9:
        return "four"
    elif n == -1:
        return "Finish It!"
    else:
        return "error"

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

    l = 0
    #Ignore CV2 warnings
    np.seterr(invalid='ignore')

    #print 'Trying to Connect to java Server'
    #connect_To_Server()

    #List Count Detected Gesture
    count = [0 for i in range(10)]

    ans = raw_input('Do u have a plain background or not(y/n)? ')
    if ans == 'y':
        fgbg = cv2.BackgroundSubtractorMOG2(history=1,varThreshold=50,bShadowDetection=True)
    else:
        ##custom made background subtraction
        bgs,ch = BackgroundSubtractorSuperMOG()

    # Initialize windows and trackbars
    createWindows()
    addTrackbars()

    adjustImageFlag = True
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame2 = frame.copy()

        # Get a nice and clean YCC image to work on
        # TODO: BG subtraction
        #####background subtraction starts#####
        if ans == 'y':
            mask = fgbg.apply(frame2)
            cv2.imshow('SubtractedBackground',mask)
            frame2 = cv2.bitwise_and(frame2,frame2,mask=mask)
        else:
            mask = applyChoice(ch,frame,bgs)
            cv2.imshow('SubtractedBackground',mask)
            frame2 = cv2.bitwise_and(frame2,frame2,mask=mask)
        #####done#####

        imgYCC = transform_image(frame2)


        if not adjustImageFlag:
            # Get contours of the hand
            handContour = getHandContours(imgYCC)
            if handContour is None:
                print "No hand contour exception +++++++++++++++++++++++++++++++++++++++++++++++++++"
                continue
            #print '**********************************************************************************************'
            #print handContour
            #print '**********************************************************************************************'


            fingertips, center, rad, handDimens = detectHand(handContour)

            if fingertips is None or center is None:
                continue

            cv2.circle(frame2, tuple(center), 5, [0, 0, 0], 2)
            cv2.circle(frame2, tuple(center), int(rad), [50, 1, 164], 2)

            for tip in fingertips:
                cv2.circle(frame2, tuple(tip), 4, [18, 123, 251], 2)
                cv2.line(frame2, tuple(tip), tuple(center), [245, 157, 100], 2)


            ######message to send to server#########
            detected_gesture = detectGesture(fingertips, center, handDimens)
            if detected_gesture == "open-hand":
                count[0] += 1
            elif detected_gesture == "closed-hand":
                count[1] += 1
            elif detected_gesture == "fist":
                count[2] += 1
            elif detected_gesture == "pointing-x":
                count[3] += 1
            elif detected_gesture == "pointing-y":
                count[4] += 1
            elif detected_gesture == "metal":
                count[5] += 1
            elif detected_gesture == "gun":
                count[6] += 1
            elif detected_gesture == "two":
                count[7] += 1
            elif detected_gesture == "three":
                count[8] += 1
            elif detected_gesture == "four":
                count[9] += 1
            elif detected_gesture == "Finish It!":
                return "Finish It!"
            else:
                return "error"

            if l == 10:
                temp = count.index(max(count))
                detected_Gesture = maxGestureDetected(temp)
                if type(detected_Gesture) is type('anyString'):
                    print type(detected_Gesture)
                    #sendQueryToServer(detected_Gesture)
                l = 0
                count = [0 for i in range(10)]

            print detected_gesture
            l += 1


        if frame.any():
            cv2.imshow('YCCCapture',imgYCC)
            cv2.imshow('Output',frame2)


        k = cv2.waitKey(10)
        if k == 27 or k == ord('q'):
            break
        if k == 32:
            adjustImageFlag = False
            print adjustImageFlag

    # Cleanup
    cap.release()
    del(cap)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
