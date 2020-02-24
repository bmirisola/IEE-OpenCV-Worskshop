import numpy as np
import cv2
import Constants

#Set up camera
cap = cv2.VideoCapture(Constants.CAPTURE_SOURCE_ID)

#define function to print pixel values on BGR frame with Left Button Click
def print_bgr_at_coord(event, x, y,empty, data):
    global frame
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(frame[y, x])

#call a blank window frame
cv2.namedWindow('frame')

#set print bgr value function to fame window
cv2.setMouseCallback('frame', print_bgr_at_coord)

# NumPy to create arrays to hold lower and upper range
# The “dtype = np.uint8” means that data type is an 8 bit integer
lower_range = np.array([18, 100, 100], dtype=np.uint8)
upper_range = np.array([38, 255, 255], dtype=np.uint8)

while(True):
    #set frame variable equal to camera input
    ret, frame = cap.read()
	
	# convert BGR image to a HSV image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
    # convert BGR image to a HSV image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
	# create a mask for image
    mask = cv2.inRange(hsv, lower_range, upper_range)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red rectangle on frame
        cv2.drawContours(frame, [box], 0, (0, 0, 255))

        # finally, get the min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        # convert all values to int
        center = (int(x), int(y))
        radius = int(radius)
        # and draw the circle in blue
        frame = cv2.circle(frame, center, radius, (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    cv2.imshow('hsv', hsv)
    cv2.imshow('mask', mask)

	#stop running when q is hit. Checks every 100 milliseconds
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()