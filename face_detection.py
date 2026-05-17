# FACE DETECTION
import cv2
#LOAD HAAR CASCADE
face_cascade=cv2.CascadeClassifier(
    cv2.data.haarcascades+'haarcascade_frontalface_default.xml'
)
eye_cascade=cv2.CascadeClassifier(
    cv2.data.haarcascades+'haarcascade_eye.xml'
)
#START WEBCAM
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break
    #CONVERT TO GRAYSCALE
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #DETECT FACES
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        #DRAW RECTANGLE AROUND THE FACE
        cv2.rectangle(frame,(x,y),(x+w,y+h),(168, 50, 164),2)
        #REGION OF INTREST (ROI) FOR EYES INSIDE FACE
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        #DETECT EYES INSIDE FACE
        eyes=eye_cascade.detectMultiScale(roi_gray,1.3,5)
        for(ex,ey,ew,eh)in eyes:
            #DRAW RECTANGLE AROUND EYES(BLUE)
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    # SHOW OUTPUT
    cv2.imshow("Face & Eye Detection",frame)
    # EXIT ON'q'
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
