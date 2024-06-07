import cv2
import numpy as np
from PIL import Image
import os

def train_model():
    data_dir = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    face_samples = []
    ids = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith("jpg"):
                path = os.path.join(root, file)
                id_ = int(os.path.basename(root))
                img = Image.open(path).convert("L")  # convert to grayscale
                img_numpy = np.array(img, "uint8")
                faces = face_cascade.detectMultiScale(img_numpy)
                
                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y+h, x:x+w])
                    ids.append(id_)
    
    recognizer.train(face_samples, np.array(ids))
    recognizer.write('trained_data.yml')

if __name__ == "__main__":
    train_model()
