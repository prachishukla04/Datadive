#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install face-recognition


# In[2]:


pip install opencv-python


# In[3]:


import face_recognition as fr
import numpy as np
import cv2
import pandas as pd
from datetime import datetime 

# Load known face image and encoding
known_face_image = fr.load_image_file(r"C:\Users\Prachi\Downloads\prachi.jpeg")
known_face_encoding = fr.face_encodings(known_face_image)[0]
known_face_encodings = [known_face_encoding]

known_face_names = ["PRACHI"]

#initialize attendance dictionary

attendance={name:False for name in known_face_names}

video_capture = cv2.VideoCapture(0)

while True:
    
    ret, frame = video_capture.read()
    rgb_frame=frame[:,:,::-1]
   
    face_locations = fr.face_locations(frame)
    face_encodings = fr.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known face
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            attendance[name]=True

        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
attendance_records = []
timestamp = datetime.now().strftime('%y-%m-%d %H:%M:%S')
for name, present in attendance.items():
    if present:
        attendance_records.append({"Name": name, "Timestamp": timestamp})

attendance_df = pd.DataFrame(attendance_records)
attendance_df.to_excel("attendance.xlsx", index=False)
       
# Release the webcam and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




