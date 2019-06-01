import cv2
import sys

""" Turns passed image into a cascade """
faceCasc = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')
eyeCasc = cv2.CascadeClassifier('classifiers/haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)

while True:
    """ vid_frame is one frame of the webcam feed
        frame_rem is the number of frames remaining in the feed (only matters for video, not webcam) """
    frame_rem, vid_frame1 = video_capture.read()

    """ Resize the video frame to increase fps """
    h, w, lay = vid_frame1.shape
    vid_frame = cv2.resize(vid_frame1, (w // 2, h // 2))

    """ Converts vid_frame into a grayscale image """
    grayscale = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)

    faces = faceCasc.detectMultiScale(
        grayscale,
        scaleFactor=1.2,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(30, 30),
        #maxSize=(100,100)
    )

    eyes = eyeCasc.detectMultiScale(
        grayscale,
        scaleFactor=1.2,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(30, 30),
        # maxSize=(100,100)
    )

    if len(eyes) > 0:
        print('at least 1 eye opened')
    else:
        print('both eyes closed')


    """ Draws rectangle around faces in green and eyes in red"""
    for (x, y, w, h) in faces:
        cv2.rectangle(vid_frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    for (x, y, w, h) in eyes:
        cv2.rectangle(vid_frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

    """ Display resulting frame """
    cv2.imshow('Video', vid_frame)

    """ Break if 'q' is pressed """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

""" Release capture """
video_capture.release()
cv2.destroyAllWindows()
