import torchvision.models as TM
import torchvision.transforms as T
import torch
import cv2
import time
class HumanDetection:
    # define variables
    model = None
    COCO_CLASSES = []

    # default constructor
    def __init__(self):
        self.model = TM.detection.ssdlite320_mobilenet_v3_large(weights="SSDLite320_MobileNet_V3_Large_Weights.DEFAULT")
        #self.model = TM.detection.ssdlite320_mobilenet_v3_large(weights="SSDLite320_MobileNet_V3_Large_Weights.DEFAULT")
        #self.model = TM.detection.ssd300_vgg16(pretrained=True)
        self.model.eval()
        # Class labels for the COCO dataset, will be added to a txt file later
        self.COCO_CLASSES = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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
    def runDetection(self):
        transformtotensor = T.ToTensor()
        video = cv2.VideoCapture(0)

        # Testing seconds to get to 30 frames
        start = time.time()
        # image processing code... unexpanded

        while True:

            # read from video webcam
            ret, frame = video.read()
            # break loop if reading from frame is not successful
            if not ret:
                break

            # If process this frame is set to true:
            #if processing_current_frame:
            # transform the frame to tensor for object detection
            tframe = transformtotensor(frame).unsqueeze(0)

            # filter all gradients, perform detection
            with torch.no_grad():
                detections = self.model(tframe)

            # get the bounding boxes, scores and labels
            bbox,scores,labels = detections[0]['boxes'], detections[0]['scores'], detections[0]['labels']
            nums = torch.argwhere(scores > 0.8).shape[0]

            for i in range(nums):

                # filter all objects, detect only people
                if labels[i].item() == 1:
                    x, y, w, h = bbox[i].numpy().astype('int')
                    cv2.rectangle(frame, (x,y), (w,h), (0,0,255),5)
                    cv2.putText(frame, str(self.COCO_CLASSES[labels[i].item()]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            # Display the frame
            cv2.imshow('Webcam', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # stop the timer
        end = time.time()
        # get the number of seconds to reach 30 frames
        seconds = end - start
        print("Time taken to get 30 frames for SSD300_VGG16: {0} seconds".format(seconds))
        print("Frames per second: {0}".format(30/seconds))
        # Release the webcam and close windows
        video.release()
        cv2.destroyAllWindows()

# Run the detection code
det = HumanDetection()
det.runDetection()