import cv2
import numpy as np
from tensorflow.keras.models import load_model
import turtle
import smtplib
from email.mime.text import MIMEText

model = load_model('model.h5')

def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def display_result(is_compliant):
    screen = turtle.Screen()
    screen.title("Uniform Inspector")
    pen = turtle.Turtle()
    pen.hideturtle()
    pen.speed(0)
    pen.pensize(10)
    pen.penup()
    pen.goto(0, 0)
    pen.pendown()

    if is_compliant:
        pen.pencolor("green")
        pen.right(90)
        pen.forward(50)
        pen.backward(100)
        pen.forward(50)
        pen.left(45)
        pen.forward(70)
    else:
        pen.pencolor("red")
        pen.right(45)
        pen.forward(100)
        pen.backward(200)
        pen.forward(100)
        pen.right(90)
        pen.forward(100)
        pen.backward(200)

    turtle.done()
def send_email(student_name, subject, body):
    sender = 'rajibul45@outlook.com'
    receiver = 'jinatpervin98@gmail.com'
    subject = 'Uniform Violation'
    body = f'Student {student_name} is wearing a wrong uniform.'


    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    try:
        with smtplib.SMTP('smtp.office365.com', 587) as server:
            server.starttls()
            server.login(sender, 'nghgnboihcxxswvt')
            server.sendmail(sender, receiver, msg.as_string())
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")


def main():

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trained_data.yml")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            id_, _ = recognizer.predict(roi_gray)
            if id_ == 1:  # Replace 1 with the ID of the student
                student_name = "Student Name"  # Provide the name corresponding to the ID
                break
        else:
            print("No recognized faces found.")
            return

        processed_frame = preprocess_image(frame)
        prediction = model.predict(processed_frame)
        class_label = np.argmax(prediction)

        if class_label == 1:
            is_compliant = True
            print("Uniform is compliant.")
            send_email(student_name, 'Uniform CHECK', f'Student {student_name} is wearing a uniform.')
        else:
            is_compliant = False
            print("Uniform is non-compliant.")
            send_email(student_name, 'Uniform Violation', f'Student {student_name} is wearing a wrong uniform.')

        display_result(is_compliant)
    else:
        print("Failed to capture image.")

if __name__ == "__main__":
    main()
