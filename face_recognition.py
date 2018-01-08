import cv2

# Load the cascade. Add more cascade to detect more things, the link is below:
# https://github.com/opencv/opencv/tree/master/data/haarcascades
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('mouth.xml')

"""
Face recognition function using opencv

Args:
    - gray_frame: frame/image in gray scale
    - frame: the original color frame

Return: the same color frame with the detector box around
    
"""
def detect(gray_frame, frame):
    # detect person's face and draw a blue box around - scaleFactor and minNeighbors can be adjusted
    faces = face_cascade.detectMultiScale(image=gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img=frame, pt1=(x,y), pt2=(x+w, y+h), color=(255,0,0), thickness=2)
        face_gray = gray_frame[y:y+h, x:x+w] # area of the face
        face_color = frame[y:y+h, x:x+w] # area of the face
        
        # detect the eyes and draw green boxes around them - scaleFactor and minNeighbors can be adjusted
        eyes = eye_cascade.detectMultiScale(image=face_gray, scaleFactor=1.1, minNeighbors=22)
        for (x_e, y_e, w_e, h_e) in eyes:
            cv2.rectangle(img=face_color, pt1=(x_e, y_e), pt2=(x_e+w_e, y_e+h_e), color=(0,255,0), thickness=2)
        
        # detect the mouth/smile and draw a box around it - scaleFactor and minNeighbors can be adjusted
        mouth = mouth_cascade.detectMultiScale(image=face_gray, scaleFactor=1.7, minNeighbors=22)
        for (x_m, y_m, w_m, h_m) in mouth:
            cv2.rectangle(img=face_color, pt1=(x_m, y_m), pt2=(x_m+w_m, y_m+h_m), color=(0,0,255), thickness=2)
    return frame

# start capture the video from webcam
video_capture = cv2.VideoCapture(0) # change to 1 if using external(usb) webcam
while True:
    _, frame = video_capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = detect(gray_frame, frame)
    cv2.imshow('Video', output)
    
    # press q to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()