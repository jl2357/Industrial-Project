# Links
# https://www.youtube.com/watch?v=uMzOcCNKr5A


# imports
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

# Obtain video data
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    # Convert frame to grayscale
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = model.track(frame, persist=True)

    frame_ = results[0].plot()
    # Display the video
    cv2.imshow("object", frame_)
    # press q to exit program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
