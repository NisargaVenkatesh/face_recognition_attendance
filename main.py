import cv2
import numpy as np
import face_recognition


def enhance_contrast(img):
    """Increase image contrast to improve face detection."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def detect_faces(img, img_name):
    """Detect faces, expand bounding box, and draw rectangles."""
    img = enhance_contrast(img)  # Improve visibility of facial features

    # Detect faces using CNN model for better accuracy
    faceLocs = face_recognition.face_locations(img, number_of_times_to_upsample=2, model="cnn")

    if len(faceLocs) == 0:
        print(f" No face detected in {img_name} using face_recognition!")

        # Try OpenCV Haar Cascade as backup
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print(f" No face detected in {img_name} even with OpenCV Haar Cascade!")
            return img

        print(f" OpenCV detected {len(faces)} face(s) in {img_name}.")
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)  # Draw bounding box
        return img

    faceLoc = faceLocs[0]  # Extract first detected face
    print(f" Face detected in {img_name} at location: {faceLoc}")

    # Expand bounding box for better face coverage
    expansion = 20
    top = max(0, faceLoc[0] - expansion)
    right = min(img.shape[1], faceLoc[1] + expansion)
    bottom = min(img.shape[0], faceLoc[2] + expansion)
    left = max(0, faceLoc[3] - expansion)

    # Draw expanded rectangle
    cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 255), 2)

    return img


# Load images
imgElon = face_recognition.load_image_file('ImagesBasic/Elon-Musk.jpg')
imgTest = face_recognition.load_image_file('ImagesBasic/BillGates.jpg')

# Detect and process faces in images
imgElon = detect_faces(imgElon, "Elon-Musk.jpg")
imgTest = detect_faces(imgTest, "BillGates.jpg")

# Encoding images
encodeElon = face_recognition.face_encodings(imgElon)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]

# Comparing encodings to match, backend we use linear SVMs
results = face_recognition.compare_faces([encodeElon], encodeTest)
print(results)

# Comparing distances, lower the distance, better the match.
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(faceDis)

# to add this text on the image
cv2.putText(imgTest, f'{results, round(faceDis[0],2)}',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

# Show processed images
cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Musk Test', imgTest)
cv2.waitKey(0)
cv2.destroyAllWindows()

