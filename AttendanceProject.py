import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'

# creating the list of all the images we will import
images = []
names = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    names.append(os.path.splitext(cl)[0])
print(names)


def findencodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')


encodeListKnown = findencodings(images)
print('Encoding complete')

cap = cv2.VideoCapture(0)

try:
    while True:
        success, img = cap.read()
        imgsm = cv2.resize(img, (0,0), None, 0.25, 0.25)
        imgsm = cv2.cvtColor(imgsm, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgsm)
        encodesCurFrame = face_recognition.face_encodings(imgsm, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = names[matchIndex].upper()
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y1-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                markAttendance(name)

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)

        # Press 'q' to exit smoothly
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n Exiting...")
            break

except KeyboardInterrupt:
    print("\n Program interrupted manually")

finally:
    print(" Releasing resources")
    cap.release()  # Release webcam
    cv2.destroyAllWindows()  # Close OpenCV windows
    print(" Clean exit completed")