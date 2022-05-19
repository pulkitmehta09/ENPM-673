import numpy as np
import cv2
import argparse
from math import sqrt, pow

def dist(x1, y1, x2, y2):
    return sqrt(pow(x1-x2, 2) + pow(y1-y2, 2))

def get_speed(ox, oy, nx, ny, lane=0):
    # print(f"measuring speed between {measurement_region[lane-1][0][1]} and {measurement_region[lane-1][1][1]}")
    if (oy > measurement_region[lane][0][1] and oy < measurement_region[lane][1][1] and 
        ox < measurement_region[lane][2][0] and ox > measurement_region[lane][1][0]):
        pixeldistance = dist(ox, oy, nx, ny)
        pixelspersecond = pixeldistance * fps
        meterspersecond = pixelspersecond / pixelspermeter
        # return kmph
        return round(meterspersecond * 3.6, 1)
    else:    
        return 0

def resetlanespeed():
    for i in range(len(lanespeed)):
        lanespeed[i] = 0
    

def warp(frame):
    height, width = frame.shape[:2]
    # pts1 = np.float32([[(145, height), (580,330), (725,330), (1170, height)]])
    # pts2 = np.float32([[0,400],[0,0],[200,0],[200,400]])
    pts1 = np.float32([[(540, 450), (610,400), (710,400), (770, 450)]])
    pts2 = np.float32([[0,400],[0,0],[200,0],[200,400]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(frame, matrix, (200,400))

    return warped, matrix

measurement_region = [
    np.array([[ 10,  309], [ 10, 383] ,[ 195, 309], [ 195,  383]]),
    np.array([[ 10,  309] ,[ 195, 309], [ 195,  383], [ 10, 383]])]

lanespeed = [0]

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.1,
                       minDistance = 50,
                       blockSize = 3 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


video_file = "highway5.mp4"
cap = cv2.VideoCapture(video_file)

# Constants
lanemarkingpixel = 377-306
pixelspermeter = lanemarkingpixel / 3.048
fps = round(cap.get(cv2.CAP_PROP_FPS))
print(f"FPS: {fps}")


# Take first frame and find corners in it
ret, old_frame = cap.read()
roi_mask = old_frame
roi_mask[:, :540] = 0
roi_mask[:, 760:] = 0
roi_mask[:380, :] = 0
roi_mask[520:, :] = 0

old_gray = cv2.cvtColor(roi_mask, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

warped_old_gray, _ = warp(old_gray)

height, width = old_frame.shape[:2]

# Create a mask image to put visual indicators
warped_old_frame, _ = warp(old_frame)
mask = np.zeros_like(warped_old_frame)

feature_refresh_counter = 0

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    
    frame[:, :540] = 0
    frame[:, 760:] = 0
    frame[:380, :] = 0
    frame[520:, :] = 0

    warped_bgr_frame, _ = warp(frame)

    # speed cam regions
    cv2.fillPoly(mask, [measurement_region[1]], (0,0,50))

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    warped_gray_frame, _ = warp(gray_frame)

    # recalculate good features
    if feature_refresh_counter % 20 == 0:
        p0 = cv2.goodFeaturesToTrack(warped_old_gray, mask=None, **feature_params)


    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(warped_old_gray, warped_gray_frame, p0, None, **lk_params)
    
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    # else:
    #     print('None')
    
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        
        speed = get_speed(a, b, c, d, lane=0)
        if speed:
            if speed > 1:
                lanespeed[0] = speed

        print("{:4}".format(lanespeed[0]))
        mask = cv2.circle(mask, (int(a), int(b)), 5, (0,255,0), -1)
        mask = cv2.circle(mask, (int(c), int(d)), 5, (0,0,255), -1)
    img = cv2.add(warped_bgr_frame, mask)
    # img = frame
    
    cv2.imshow('frame', img)

    key = cv2.waitKey(fps)
    if key == ord("q"):
        break
    
    
    # Now update the previous frame and previous points
    warped_old_gray = warped_gray_frame.copy()
    p0 = good_new.reshape(-1, 1, 2)
    feature_refresh_counter += 1
    mask = np.zeros_like(warped_old_frame)


    
cv2.destroyAllWindows()