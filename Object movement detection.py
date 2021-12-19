import cv2
import imutils.video
import datetime
import time


video = imutils.video.VideoStream(src=0).start()
time.sleep(2.0)
first_frame = None

while True:
    vframe = video.read()
    text = "Nothing is moving"

    if vframe is None:
        break

    # modify the frame to increase the accuracy of the calculation of the difference
    frame = imutils.resize(vframe, width=500)
    gray = cv2.cvtColor(vframe, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    # initialize the first frame - it should be a static background
    if first_frame is None:
        first_frame = blur
        continue

    difference_frame = cv2.absdiff(first_frame, blur)
    # second item of the tuple - the actual frame
    threshold = cv2.threshold(difference_frame, 25, 255, cv2.THRESH_BINARY)[1]
    # dilate the threshold frame to fill the holes
    dilate = cv2.dilate(threshold, None, iterations=2)

    contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # ignore too small contours
    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        # compute and draw the bounding box on the frame
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "something's moving"

    # put webcam status and datetime on the video frame
    cv2.putText(frame, "Status: {}".format(text),
                (10, 20), cv2.FONT_ITALIC, 0.7, (255, 99, 71), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %H:%M:%S"),
                (10, frame.shape[0] - 15), cv2.FONT_ITALIC, 0.5, (255, 99, 71), 1)
    cv2.imshow("Video", frame)

    # continue to refresh the video frame
    key = cv2.waitKey(1)

# close video file or capturing device (webcam) and close any open windows
video.release()
cv2.destroyAllWindows()