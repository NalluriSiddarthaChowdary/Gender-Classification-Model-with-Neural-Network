import cv2
import numpy as np
import os
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "C:\Gender_detection\haarcascade_frontalface_default.xml")

# Prepare dataset paths (Assume you have images stored in 'male/' and 'female/' folders)
dataset_path = "C:/Gender_detection/train"
categories = ["Male", "Female"]

1.3
data = []
labels = []

# Extract HOG features from images
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))  # Resize for consistency
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return features

# Load dataset
for category in categories:
    folder_path = os.path.join(dataset_path, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            features = extract_features(img)
            data.append(features)
            labels.append(category)

# Convert labels to numeric values
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# Train SVM model
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
svm_classifier = SVC(kernel="linear", probability=True)
svm_classifier.fit(X_train, y_train)

# Open webcam for real-time gender detection
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_features = extract_features(face)
        face_features = np.array(face_features).reshape(1, -1)

        # Predict gender
        gender_index = svm_classifier.predict(face_features)[0]
        gender_label = encoder.inverse_transform([gender_index])[0]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, gender_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Real-Time Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
