import numpy as np
import cv2
import argparse
from math import sqrt, pow
import imutils

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
    return 0

def warp(frame):
    height, width = frame.shape[:2]
    pts1 = np.float32([[(1030, 615), (260,395), (330,375), (1180, 485)]])
    pts2 = np.float32([[0,400],[0,0],[200,0],[200,400]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(frame, matrix, (200,400))

    return warped, matrix


def resetlanespeed():
    for i in range(len(lanespeed)):
        lanespeed[i] = 0

video_file = "custom.mp4"
# video_file = "highway2.mp4"
# video_file = "highway3.mp4"
# video_file = "highway4.mp4"
# video_file = "highway5.mp4"



measurement_region = [
    np.array([[0,  290], [0, 330], [80, 330], [80,  290]]),
    
    ]

lanespeed = [0, 0, 0]

cap = cv2.VideoCapture(video_file)

# Constants
lanemarkingpixel = 30
pixelspermeter = lanemarkingpixel / 3.048
fps = round(cap.get(cv2.CAP_PROP_FPS))
print(f"FPS: {fps}")
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 20,
                       blockSize = 3 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

warped_old_gray, _ = warp(old_gray)

height, width = old_frame.shape[:2]

# check corners
# for i in p0:
#     x, y = i.ravel()
#     old_frame = cv2.circle(old_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
# cv2.imshow("old_frame",old_frame)
# cv2.waitKey(0)

# Create a mask image for drawing purposes
warped_old_frame, _ = warp(old_frame)
mask = np.zeros_like(warped_old_frame)


feature_refresh_counter = 0

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    
    frame = imutils.resize(frame,width = 1200)
    
    frame_copy = frame.copy()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]    
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20,20))
    cl1 = clahe.apply(v)
    hsv[:,:,2] = cl1
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    clahe_img = frame.copy()
    
    warped, matrix = warp(frame)

    # speed cam regions
    cv2.fillPoly(mask, [measurement_region[0]], (0,0,50))
    # cv2.fillPoly(mask, [measurement_region[1]], (0,0,50))
    # cv2.fillPoly(mask, [measurement_region[2]], (0,0,50))
    

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    warped_frame_gray, _ = warp(frame_gray)
    # recalculate good features
    if feature_refresh_counter % 20 == 0:
        p0 = cv2.goodFeaturesToTrack(warped_old_gray, mask=None, **feature_params)
        # mask = np.zeros_like(old_frame)


    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(warped_old_gray, warped_frame_gray, p0, None, **lk_params)
    
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
       
        for i in range(1):
            speed = get_speed(a, b, c, d, lane=i)
            if speed:
                if speed > 1:
                    lanespeed[i] = speed
            

        print("{:4}"
          .format(lanespeed[0]))

        # mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        # mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0,255,0), 2)
        # warped = cv2.circle(warped, (int(a), int(b)), 5, (0,255,0), -1)
        # warped = cv2.circle(warped, (int(c), int(d)), 5, (0,0,255), -1)
    img = cv2.add(warped, mask)
    # img = frame
    cv2.imshow('img', img)

    cv2.putText(frame, 'Lane1: {:.0f} km/h'.format(lanespeed[0]), (575,650), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 2)    
    # cv2.putText(frame, 'Lane2: {:.0f} km/h'.format(lanespeed[1]), (705,650), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 2)    
    # cv2.putText(frame, 'Lane3: {:.0f} km/h'.format(lanespeed[2]), (975,650), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 2)    

    # cv2.imshow('frame', frame)
    unwarped_mask = cv2.warpPerspective(mask, np.linalg.inv(matrix), (frame.shape[1],frame.shape[0]))

    result = cv2.add(unwarped_mask,frame)

    cv2.imshow('result', result)
    # cv2.imshow('original', frame_copy)
    # cv2.imshow('clahe', clahe_img)

    key = cv2.waitKey(fps)
    if key == ord("q"):
        break
    
    resetlanespeed()
    
    # # Now update the previous frame and previous points
    warped_old_gray = warped_frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    feature_refresh_counter += 1
    

cv2.destroyAllWindows()