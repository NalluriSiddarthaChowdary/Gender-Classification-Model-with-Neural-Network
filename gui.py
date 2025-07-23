import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from playsound import playsound
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import threading

# Load CNN model
model_path = r"C:/Gender_detection/gender_detection.model"
model = load_model(model_path)
input_shape = model.input_shape[1:3]
input_channels = model.input_shape[-1]
GENDER_CLASSES = ["Male", "Female"]
SOUND_EFFECTS = {
    "Male": "male_sound.mp3",
    "Female": "female_sound.mp3"
}

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Prediction function
def predict_gender(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    for (x, y, w, h) in faces:
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, input_shape)
        if input_channels == 1:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = np.expand_dims(roi, axis=-1)
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=0)
        prediction = model.predict(roi)
        gender_index = np.argmax(prediction[0])
        gender = GENDER_CLASSES[gender_index]
        confidence = prediction[0][gender_index]
        try:
            playsound(SOUND_EFFECTS[gender], block=False)
        except:
            pass
        label = f"{gender} ({confidence * 100:.1f}%)"
        color = (0, 255, 0) if gender == "Male" else (255, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img

# GUI setup
class GenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gender Guardian")
        self.root.configure(bg="#B388EB")
        self.root.geometry("800x600")

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.upload_btn = tk.Button(root, text="Upload Image", font=("Arial", 14), command=self.upload_image)
        self.upload_btn.pack(pady=10)

        self.live_btn = tk.Button(root, text="Start Live Detection", font=("Arial", 14), command=self.toggle_webcam)
        self.live_btn.pack(pady=10)

        self.running = False
        self.cap = None

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if path:
            img = cv2.imread(path)
            result_img = predict_gender(img.copy())
            self.show_image(result_img)

    def toggle_webcam(self):
        if not self.running:
            self.running = True
            self.live_btn.config(text="Stop Live Detection")
            threading.Thread(target=self.start_webcam, daemon=True).start()
        else:
            self.running = False
            self.live_btn.config(text="Start Live Detection")

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not access webcam.")
            return
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            result_img = predict_gender(frame.copy())
            self.show_image(result_img)
        self.cap.release()
        self.image_label.config(image='')

    def show_image(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((700, 500))
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)

# Launch GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = GenderApp(root)
    root.mainloop()
