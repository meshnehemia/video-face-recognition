import cv2
import numpy as np
import face_recognition as face_rec
import os

# Load known faces from the images database
path = 'images_database'
known_faces = []
known_names = []

for cl in os.listdir(path):
    img_path = os.path.join(path, cl)
    img = face_rec.load_image_file(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_rec.face_encodings(img)[0]
    known_faces.append(encode)
    known_names.append(os.path.splitext(cl)[0])

# Open video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    resized_frame = cv2.resize(frame, (640, 480))
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Detect face locations and encodings
    face_locations = face_rec.face_locations(rgb_frame)
    face_encodings = face_rec.face_encodings(rgb_frame, face_locations)

    # Process each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Draw rectangle around the face
        cv2.rectangle(resized_frame, (left, top), (right, bottom), (255, 0, 255), 2)

        # Compare with known faces
        matches = face_rec.compare_faces(known_faces, face_encoding)
        name = 'Not Recognized'

        # Find the best match
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        # Draw label below the face
        cv2.putText(resized_frame, name, (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', resized_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()