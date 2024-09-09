# Resources
# https://www.youtube.com/watch?v=DfVsXMkv8cA : to install the imports

# https://www.youtube.com/watch?v=V62M9d8QkYM&t=62s : the opencv code
# Note the link above, detectCommonObjects doesn't work

# https://www.youtube.com/watch?v=88HdqNDQsEk : using the Haar Cascade
# Note the link above works and can detect faces

# Imports
import cv2
import time
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Obtain video data
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces using haar cascade file
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Identify faces using a rectangle and square
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 5)
        cv2.putText(frame, 'Human Identified!', (x+w,y), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
        
    # Display the video
    cv2.imshow("object", frame)
    # press q to exit program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
