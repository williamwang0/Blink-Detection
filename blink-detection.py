import cv2


#FUNCTIONS
def eyeInFace(e, f):
    """ Returns true if eye is contained within a detected face """
    for (fx, fy, fw, fh) in f:
        if fx < e[0] < fx + w and fy < e[1] < fy + h:
            return True
    return False


""" Turns passed image into a cascade """
faceCasc = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')
eyeCasc = cv2.CascadeClassifier('classifiers/haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)

# blinked boolean to make sure one long blink is still counted as one blink
blinked = True
blink_count = 0

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
        scaleFactor=1.3,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(30, 30)
    )

    eyes = eyeCasc.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=7,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(30, 30)
    )

    """ Filters out eyes so that all eyes detected are contained inside a face """
    eyes = [eye for eye in eyes if eyeInFace(eye, faces)]

    """ Counts how many times someone blinks in the feed """
    if len(eyes) == 0:
        if blinked:
            blink_count += 1
            print('blinked!')
            print('You blinked ' + str(blink_count) + ' times so far')
            blinked = False
        pass
    else:
        blinked = True

    """ Draws rectangle around faces in green and eyes in red """
    for (x, y, w, h) in faces:
        cv2.rectangle(vid_frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    for (x, y, w, h) in eyes:
        cv2.rectangle(vid_frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

    """ Display resulting frame """
    cv2.imshow('Video', vid_frame)

    """ Break if 'q' is pressed """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

""" Release capture """
video_capture.release()
cv2.destroyAllWindows()
