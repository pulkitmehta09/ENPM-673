import numpy as np
import cv2
import argparse

# video_file = "highway1.mp4"
# video_file = "highway2.mp4"
# video_file = "highway3.mp4"
# video_file = "highway4.mp4"
video_file = "highway5.mp4"


cap = cv2.VideoCapture(video_file)




# Create a mask image for drawing purposes

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
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshframe = cv2.threshold(gray_frame, 210, 255, cv2.THRESH_BINARY)

    graythresh = threshframe
    print(threshframe[600])

    cv2.imshow('frame', graythresh)

    key = cv2.waitKey(30)
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()