import cv2
import sys

""" Turns passed image into a cascade """
cascPath = sys.argv[1]
faceCasc = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    """ vid_frame is one frame of the webcam feed
        frame_rem is the number of frames remaining in the feed (only matters for video, not webcam) """
    frame_rem, vid_frame = video_capture.read()

    """ Converts vid_frame into a grayscale image """
    grayscale = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)

    faces = faceCasc.detectMultiScale(
        grayscale,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    """ Draws rectangle around faces """
    for (x, y, w, h) in faces:
        cv2.rectangle(vid_frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    """ Display resulting frame """
    cv2.imshow('Video', vid_frame)

    """ Break if 'q' is pressed """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

""" Release capture """
video_capture.release()
cv2.destroyAllWindows()
