import math
import numpy as np

def detectGesture(fingers, centroid, handDimens):
    orientation = 0 if handDimens[1] > handDimens[0] else 1
    n = len(fingers)
    angles = getAngles(fingers, centroid)
    if n > 2 and all(angle < 20 for angle in angles):
        return "closed-hand"
    if n == 0:
        return "fist"
    elif n == 1:
        # TODO: Detect finger based on relative position in box
        if orientation:
            return "pointing-y"
        else:
            return "pointing-x"
    elif n == 2:
        # TODO: Detect finger based on relative position in box
        if 70 < angles[0] < 105:
            return "gun"
        elif 55 < angles[0] < 70:
            return "metal"
        else: return "two"
    elif n == 3:
        return "three"
    elif n == 4:
        return "four"
    else:
        return "open-hand"

def getAngles(fingers, centroid):
    angles = []
    for i in range(len(fingers)-1):
        lens = np.square([dist(fingers[i], centroid), dist(fingers[i+1], centroid), dist(fingers[i], fingers[i+1])])
        a = np.rad2deg(np.arccos((lens[0] + lens[1] - lens[2])/(2*dist(fingers[i], centroid)*dist(fingers[i+1], centroid))))
        angles.append(a)
    return angles

# Calculates distance between two points
def dist(point, center):
    return math.hypot(point[0] - center[0], point[1] - center[1])
