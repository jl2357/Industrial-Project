# Imports
import os
import sys
import math
import numpy
import cv2
import face_recognition

# Obtain the percentage of face match
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

# Human recognition using face recognition package
class HumanRecognition:
    # Define variables
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    # Default constructor
    def __init__(self):
        self.encode_faces()

    # Facial detection
    def encode_faces(self):
        for image in os.listdir('Saved_people'):
            # Find faces in the saved_people folder. if no faces identified, an error is returned
            face_image = face_recognition.load_image_file(f'Saved_people/{image}')

            try:
                # Encode faces in the Saved_people folder to use for facial recognition
                face_encoding = face_recognition.face_encodings(face_image)[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(image)
                print(self.known_face_names)
            except:
                # Catch the error created if no faces are found
                # If no faces are identified in a file (From taking an image), remove the file
                print("Unable to identify face in %s, removing file" % image)
                os.remove(f'Saved_people/{image}')

    def run_recognition(self):
        # Obtain video data
        video = cv2.VideoCapture(0)
        # Exit if no video source is found
        if not video.isOpened():
            sys.exit('Unable to display video')

        while True:
            # Read from the video capture
            ret, frame = video.read()
            # Processing every other frame to save resources. If process_current_frame is set to true, run code
            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                rgb_small_frame = numpy.ascontiguousarray(small_frame[:, :, ::-1])

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []

                # Go through all face encodings
                for face_encoding in self.face_encodings:
                    # Compare the faces in face encoding with identified face
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    # Set default value of all identified faces as Unknown
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = numpy.argmin(face_distances)

                    # If a match has been found
                    if matches[best_match_index]:
                        # Get file name and confidence level of identified face
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    # If no matches are found, save a new face image to Saved_people for later identification
                    elif name == 'Unknown':
                        # Create the fiel name for the newly saved image
                        # Add file names saved as ID, incrementing ID if already exists in file
                        i = 0
                        while os.path.exists('Saved_people/Person_ID%s.png' % i):
                            i += 1

                        # Take a picture of the unidentified face, and save it to the Saved_people folder
                        cv2.imwrite('Saved_people/Person_ID%s.png' % i, frame)
                        # Re-encode the face of the saved file
                        self.encode_faces()

                    self.face_names.append(f'{name} ({confidence})')

            # Change the state of process_current_frame, only process every other frame to save resources
            self.process_current_frame = not self.process_current_frame

            # Display bounding boxes
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                # Draw rectangles and add the confidence level
                cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255, 1))
            cv2.imshow('Human recognition', frame)

            # Break loop if user presses q
            if cv2.waitKey(1) == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()


# Testing the facial recognition
test = HumanRecognition()
test.run_recognition()