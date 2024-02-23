from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# frame = cv2.imread("2.png")
cap = cv2.VideoCapture('bad.avi')

objects = []
finalContour = None
pixelsPerMetric = None
switch = None


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def mouse_click(event, x, y,  flags, param):
    global objects
    if event == cv2.EVENT_LBUTTONDOWN:
        objects.append([x, y])
        print(objects)


cv2.namedWindow('frame')
cv2.setMouseCallback('frame', mouse_click)


def getVideo(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.destroyWindow("Feed")
            return frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def measurement():
    global objects, finalContour, pixelsPerMetric, switch
    if switch == 1:
        (tl, tr, br, bl) = objects
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / 9
        objects = []
        print("Refrence Taken")
    else:
        cv2.drawContours(frame, [objects.astype("int")], -1, (255, 255, 0), 3)
        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = objects
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the frame
        cv2.circle(frame, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # compute the size of the object
        dimA = dA / pixelsPerMetric - 0.5
        dimB = dB / pixelsPerMetric

        # draw the object sizes on the frame
        cv2.putText(frame, "{:.1f}in".format(dimB),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 4)
        cv2.putText(frame, "{:.1f}in".format(dimB),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2)
        cv2.putText(frame, "{:.1f}in".format(dimA),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 4)
        cv2.putText(frame, "{:.1f}in".format(dimA),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2)
        return 1

# Press smallcase s to freeze video and use left mouse click to set the corner points of the shape
# Press smallcase w to establish the measurement reference (the opening squares of the black ladder on the floor)
# Press smallcase e to measure the size of the object (the white cylinder)
while True:
    frame = getVideo(cap)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("w"):
        switch = 1
    elif key == ord("e"):
        switch = 0
    else:
        continue
    objects = np.array(objects, dtype="int")
    objects = perspective.order_points(objects)
    print(objects)
    test = measurement()
    if test == 1:
        break

cv2.destroyWindow("frame")
cv2.imshow("Result", frame)
cv2.waitKey(0)
