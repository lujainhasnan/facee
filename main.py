import cv2
import pickle
import face_recognition
import cvzone
import numpy as np
import firebase_admin
from google.cloud import firestore
from google.oauth2 import service_account
from firebase_admin import credentials
from firebase_admin import storage


# Initialize the Firebase app
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': "mawjudfirebase.appspot.com"
})

# Create a credentials object from the key file
#creds = service_account.Credentials.from_service_account_file(cred)

# Initialize a Firestore client with your credentials
db = firestore.Client(project='mawjudfirebase')

bucket = storage.bucket()

# Open the camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
# print(studentIds)
print("Encode File Loaded")

counter = 0
id = -1

while True:
    success, img = cap.read()
    # Resize the image for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find the faces in the current frame
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    # Compare the current frame's encodings with the known encodings
    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                # To locate the face
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                img = cvzone.cornerRect(img, bbox, rt=0)

                # Mark the student as present
                id = studentIds[matchIndex]
                if counter == 0:
                    cv2.imshow("Face Attendance", img)
                    cv2.waitKey(1)
                    counter = 1

        if counter != 0:

            if counter == 1:
               # Get the Data
               student_ref = db.collection(u'Students').document(str(id))
               student_doc = student_ref.get()
               student_info = student_doc.to_dict()
               print(student_info)
               student_ref.update({u'attendance': u'present'})

    else:

        counter = 0

    cv2.imshow("Face Attendance", img)
    cv2.waitKey(1)
