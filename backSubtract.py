import cv2
import numpy as np

##noise removal
def removingNoise(img):
    median = cv2.GaussianBlur(img,(11,11),0)
    return median

##morphology
def morphology(frame):
    kernel = np.ones((5,5),np.uint8) #used for erorsion and dilation
    frame2 = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel) #this actually reduces noise
    return frame2

'''
##removing shadow and clearing the noise######
def ShadowRemoval(img,mask):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    img = cv2.bitwise_and(img,img,mask=mask)
    ##applying cr and cb threshold##
    y,cr,cb = cv2.split(img)
    min1 = cv2.getTrackbarPos('cr','Main')
    max1 = cv2.getTrackbarPos('CR','Main')
    min2 = cv2.getTrackbarPos('cb','Main')
    max2 = cv2.getTrackbarPos('CB','Main')
    mincrcb = np.array((0,min1,min2))
    maxcrcb = np.array((255,max1,max2))
    res = cv2.inRange(img,mincrcb,maxcrcb)
    return res
##does not fucking work right now#########
'''

def nothing(x):
    pass

ranges = []
def subtractBackground(frame,backgrounds,fromRanges = False):
    f = frame.copy()
    global ranges
    if not fromRanges:
        valueB = cv2.getTrackbarPos('diffB','M')
        valueG = cv2.getTrackbarPos('diffG','M')
        valueR = cv2.getTrackbarPos('diffR','M')
        ranges = [valueB,valueG,valueR]
    else:
        valueB = ranges[0]
        valueG = ranges[1]
        valueR = ranges[2]
    foregrounds = []


    for background in backgrounds:

        b,g,r = cv2.split(background)
        fb,fg,fr = cv2.split(frame)
        foregroundb = np.absolute(fb - b)
        foregroundg = np.absolute(fg - g)
        foregroundr = np.absolute(fr - r)
        foregroundb = foregroundb >= valueB
        foregroundg = foregroundg >= valueG
        foregroundr = foregroundr >= valueR
        foregroundb = foregroundb.astype(np.uint8)
        foregroundg = foregroundg.astype(np.uint8)
        foregroundr = foregroundr.astype(np.uint8)
        foregrounds.append([foregroundb,foregroundg,foregroundr])

    row,col = foregroundb.shape

    temp = np.zeros([row,col],dtype=np.uint8)
    temp.fill(255)

    flag = False
    for foreground in foregrounds:
        if flag == False:
            flag = True
            frameB = foreground[0]
            frameG = foreground[1]
            frameR = foreground[2]
        frameB = np.logical_and(frameB,foreground[0])
        frameG = np.logical_and(frameG,foreground[1])
        frameR = np.logical_and(frameR,foreground[2])

    frameB = frameB*temp
    frameG = frameG*temp
    frameR = frameR*temp

    frameTemp = frameG.copy()
    frameTemp = frameTemp.ravel()
    return frameB,frameG,frameR

def BackgroundSubtractorSuperMOG():

    cv2.namedWindow('M',cv2.WINDOW_NORMAL)
    cv2.namedWindow('G',cv2.WINDOW_NORMAL)
    cv2.namedWindow('R',cv2.WINDOW_NORMAL)
    cv2.createTrackbar('diffB','M',1,256,nothing)
    cv2.createTrackbar('diffG','M',1,256,nothing)
    cv2.createTrackbar('diffR','M',1,256,nothing)
    cap = cv2.VideoCapture(0)
    backgrounds = []
    count = 0
    while cap.isOpened():
        _,frame = cap.read()
        if count == 100:
            count = 0
            break
        count+=1

    while cap.isOpened():
        dump,background = cap.read()
        background = removingNoise(background.copy())
        b,g,r = cv2.split(background)
        backgrounds.append(background)
        if count == 5:
            break;
        count += 1


    while cap.isOpened():
        dump,frame = cap.read()
        frame = removingNoise(frame)
        frame2 = frame.copy()
        frameCopy0,frameCopy1,frameCopy2 = subtractBackground(frame2.copy(),backgrounds)

        cv2.imshow('G',frameCopy1)
        cv2.imshow('R',frameCopy2)

        row,col = frameCopy2.shape
        temp = np.zeros([row,col],dtype=np.uint8)
        temp.fill(255)
        new_frame = np.logical_or(frameCopy1,frameCopy2)
        new_frame = new_frame*temp
        cv2.imshow("M",new_frame)

        k = cv2.waitKey(10)
        if k == 27:
            print "Type any letter from these to chose a window(G/R/M)"
            choice = raw_input('')
            cap.release()
            del(cap)
            cv2.destroyAllWindows()
            break
    return backgrounds,choice


def applyChoice(ch,frame,backgrounds):
    frame = removingNoise(frame)
    frame2 = frame.copy()
    frameCopy0,frameCopy1,frameCopy2 = subtractBackground(frame2.copy(),backgrounds,fromRanges = True)

    if ch == 'G':
        return frameCopy1
    elif ch == 'R':
        return frameCopy2
    elif ch == 'M':
        row,col = frameCopy2.shape
        temp = np.zeros([row,col],dtype=np.uint8)
        temp.fill(255)
        new_frame = np.logical_or(frameCopy1,frameCopy2)
        new_frame = new_frame*temp
        return new_frame

'''
def main():
    bgs,ch = BackgroundSubtractorSuperMOG()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _,frame = cap.read()
        retVal = applyChoice(ch,frame,bgs)
        cv2.imshow('hello',retVal)
        k = cv2.waitKey(10)
        if k == 27:
            break

if __name__ == '__main__':
    main()
'''
