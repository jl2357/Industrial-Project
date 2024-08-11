import torchvision.models as TM
import torchvision.transforms as T
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import cv2

# Class labels for the COCO dataset (if using a COCO-pretrained model)
COCO_CLASSES = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


model = TM.detection.ssd300_vgg16(weights=ssdlite320_mobilenet_v3_large, pretrained=True)
model.eval()

# Obtain video data
video = cv2.VideoCapture(0)

transformToTensor = T.ToTensor()

while True:

    # read from video webcam
    ret, frame = video.read()
    # break loop if reading from frame is not successful
    if not ret:
        break

    # transform the frame to tensor for object detection
    tframe = transformToTensor(frame).unsqueeze(0)

    # filter all gradients, perform detection
    with torch.no_grad():
        detections = model(tframe)

    # get the bounding boxes, scores and labels
    bbox,scores,labels = detections[0]['boxes'], detections[0]['scores'], detections[0]['labels']
    nums = torch.argwhere(scores > 0.8).shape[0]

    for i in range(nums):
        print(labels[i])
        # filter all objects, detect only people
        if labels[i].item() == 1:
            x, y, w, h = bbox[i].numpy().astype('int')
            cv2.rectangle(frame, (x,y), (w,h), (0,0,255),5)
            cv2.putText(frame, str(COCO_CLASSES[labels[i].item()]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # Display the frame
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
video.release()
cv2.destroyAllWindows()