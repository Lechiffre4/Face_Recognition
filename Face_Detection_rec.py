"""This program record when a face appear in front of the webcam"""

import cv2
import time
import datetime


cap = cv2.VideoCapture(0)
frame_size = (int(cap.get(3)), int(cap.get(4)))
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

detection = False
detection_stopped_time = 0
timer_started = False
SECONDS_TO_RECORD = 5


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_file_path = input("Enter the path of the videos file : ")

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.2,5)
    bodies = body_cascade.detectMultiScale(gray,1.2,5)

    

    if (len(faces)+len(bodies))>0:
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(video_file_path+'/'+"vid_"+current_time+".mp4", fourcc,20,frame_size)
            print("rec...")
    elif detection:
        if timer_started:
            if (time.time() - detection_stopped_time) >= SECONDS_TO_RECORD:
                detection = False
                timer_started = False
                out.release()
                print("stop rec")
        else :        
            timer_started=True
            detection_stopped_time = time.time()

    if detection:
        out.write(frame)

    for(x,y,width, height) in faces:
       cv2.rectangle(frame,(x,y),(x+width, y + height),(255,0,0),2)


    cv2.imshow("Cam", frame)

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()