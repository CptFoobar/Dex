#!/usr/bin/python
import cv2
import matplotlib.pyplot as plt, math
import numpy as np
from cv2 import cv
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.qhull import QhullError

# Detects the hand from the frame. Returns coordinates of
# centroid of the hand, the finger tips and the radius of the palm circle
def detectHand(cnt):
    try:
        # FFT: Maybe we should take hand height wrt to centroid and farthest tip
        # and change logic accordingly for better and more accurate results
        minX, minY, handWidth, handHeight = cv2.boundingRect(cnt)

        # Find key points of the hand
        defectPoints, handPolygon = handKeyPoints(cnt)
        #print "handPolygon: " + str(handPolygon)

        # Calculate centroid
        centroid = (minX + handWidth*0.5, minY + handHeight*0.67)


        # Round off centroid to the nearest pixel
        centroid = np.around(centroid, decimals=1)
        centroid = centroid.astype(int)

        # Calculate radius of palm circle
        palmR = max(handWidth, handWidth)*0.35
        fingers = []
        # 1.6.. the magic factor
        magicR = palmR * 1.25

        for i in range(len(handPolygon)):
            x = dist(handPolygon[i], handPolygon[(i+1)%len(handPolygon)]) > 40
            if isFinger(handPolygon[i], centroid, handHeight, magicR) and x:
                fingers.append(handPolygon[i])

        if len(fingers) < 0 or len(fingers) > 5:
            raise ValueError

        return np.asarray(fingers), (int(handWidth*0.5), int(handHeight*0.67)), palmR, (handHeight, handWidth, minX, minY)

    except (cv2.error, QhullError, TypeError, ValueError):
        return None, None, None, None

# Calculates distance between two points
def dist(point, center):
    return math.hypot(point[0] - center[0], point[1] - center[1])

# Calculates square of distance between line and point
def distancel2p(x1,y1, x2,y2, x3,y3):
    px = x2-x1
    py = y2-y1
    dnm = px*px + py*py
    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(dnm)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3
    dy = y - y3
    dist = dx*dx + dy*dy
    return dist

def fuse(first, second):
    return (int((first[0] + second[0])/2), int((first[1] + second[1])/2))

def handKeyPoints(cnt):
    handPolygon = []
    defectPoints = []
    hull = cv2.convexHull(cnt, returnPoints = False)
    defects = cv2.convexityDefects(cnt, hull)
    try:
        s, e, f, d = defects[0, 0]
        lastEnd = tuple(cnt[s][0])
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            handPolygon.append(fuse(start, lastEnd))
            defectPoints.append(far)
            lastEnd = end
        return np.asarray(defectPoints), np.asarray(handPolygon)
    except TypeError:
        raise TypeError

def isFinger(f, c, hh, magicR):
    v = vertdist(f, c)/hh
    h = hordist(f, c)/hh
    if ((v >= 0.45 and v <= 0.75) or (h >= 0.4 and h <=0.65)) and dist(f, c) >= magicR:
        return True
    return False

# Returns the minimum distance of a point from all edges of the given polygon
def maxDistance(poly, point):
    lmin = 99999
    for i in range(len(poly)):
        x1 = poly[i][0]
        x2 = poly[(i+1)%(len(poly))][0]
        y1 = poly[i][1]
        y2 = poly[(i+1)%(len(poly))][1]
        d = distancel2p(x1, y1, x2, y2, point[0], point[1])
        if d < lmin:
            lmin = d
    return lmin

def hordist(point, ref):
    return math.fabs(point[0] - ref[0])


def vertdist(point, ref):
    return math.fabs(point[1] - ref[1])
