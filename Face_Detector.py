import cv2
import Constants

#sets contour classifier algorithm for the one classifying faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#capture's video source and sets resolution
cap = cv2.VideoCapture(Constants.CAPTURE_SOURCE_ID)
cap.set (3,640)
cap.set(4,480)

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #returns potential faces as an array of rectangles
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    #Parameters:
    #gray = frame to classify faces from
    #scaleFactor = specifies how much the image size is reduced at each image scale.
    #minNeighbors = species how many neighbors each candidate rectangle should have, to retain it. A higher number gives lower false positives.
    #minsize =  minimum rectangle size to be considered a face.


    #If faces are found, it returns the positions of detected faces as a rectangle with the left up corner (x,y)
    # and having "w" as its Width and "h" as its Height ==> (x,y,w,h).
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

    cv2.imshow('frame', frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()