import cv2

video = cv2.VideoCapture(0)
first_frame = None

while True:
    check, vframe = video.read()

    gray = cv2.cvtColor(vframe, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = blur
        continue

    difference_frame = cv2.absdiff(first_frame, blur)
    threshold = cv2.threshold(difference_frame, 25, 255, cv2.THRESH_BINARY)[1]
    dilate = cv2.dilate(threshold, None, iterations=2)

    (contours, _) = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(vframe, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "something's moving"

    cv2.imshow("Video", vframe)

    key = cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()