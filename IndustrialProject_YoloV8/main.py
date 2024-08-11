# Links
# https://www.youtube.com/watch?v=uMzOcCNKr5A
# ncnn model: https://docs.ultralytics.com/integrations/ncnn/#usage

# imports
from ultralytics import YOLO
import cv2

# getting the yolo model
model = YOLO('yolov8n.pt')

# exporting to ncnn model
model.export(format="engine")

# loading the ncnn model
trt_model = YOLO("yolov8n.engine")

# Obtain video data
video = cv2.VideoCapture(0)

results_output = []
while True:
    ret, frame = video.read()
    # Convert frame to grayscale
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = trt_model.track(frame, persist=True)
    frame_ = results[0].plot()
    # Display the video

    cv2.imshow("object", frame_)

    for result in results:
        results_output.append(result)

    # press q to exit program
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

# get output
items = []
id = []
for result in results_output:
    detection_count = result.boxes.shape[0]

    for i in range(detection_count):
        cls = int(result.boxes.cls[i].item())

        if result[i].boxes.id not in id:
            items.append(result.names[cls])
            id.append(result[i].boxes.id)

print("\nThere are " + str(len(items)+1) + " items found: ")
for i in range(len(items)):
    print("- " + str(items[i]) + ", item ID = " + str(id[i]))

video.release()
cv2.destroyAllWindows()
