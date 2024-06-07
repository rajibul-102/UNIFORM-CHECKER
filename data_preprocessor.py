import cv2
import os

def collect_images():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    student_name = input("Enter the student's name: ")
    student_id = input("Enter a numeric ID for the student: ")
    os.makedirs(f"dataset/{student_id}", exist_ok=True)
    
    count = 0
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"dataset/{student_id}/{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow('Collecting Images', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_images()
