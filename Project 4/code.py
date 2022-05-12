import cv2
from matplotlib.pyplot import gray
import numpy as np

cap = cv2.VideoCapture("highway1.mp4")
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
white = (255,255,255)

pt1 = [50, 719]
pt2 = [720, 0]
pt3 = [870, 0]
pt4 = [1279, 445]
pt5 = [1279, 719]
# _, frame = cap.read()
# print(frame.shape)
# print(cap.get(3), cap.get(4))
roi_mask = np.zeros((int(cap.get(4)),int(cap.get(3))), np.uint8)
print(type(roi_mask))
cv2.fillPoly(roi_mask, [np.array([pt1,pt2,pt3,pt4,pt5])], white)
# cv2.imshow("mask",roi_mask)
# cv2.waitKey(0)

# pts = np.array([])


while True:
    ret, frame = cap.read()

    if ret:


        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = cv2.bitwise_and(gray_frame, gray_frame, mask=roi_mask)
        # cv2.imshow("mask", roi)
        corners = cv2.goodFeaturesToTrack(roi, 100, 0.001, 5)
        corners = np.int0(corners)

        for i in corners:
            x, y = i.ravel()
            cv2.circle(frame, (x, y), 5, (0,0,255),-1)

        

        cv2.imshow("video", frame)
        key = cv2.waitKey(30)
        if key == ord("q"):
            break
    
    else: 
        break

cv2.destroyAllWindows()