#Face detection using Haar Cascades.
import cv2
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')     #Open Face Classifier.
eye = cv2.CascadeClassifier('haarcascade_eye.xml')                      #Open Eye Classifier.
image1 = cv2.imread('File Path')
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)                         #Convert to Gray Scale for Faster and Accurate results.
faces = face.detectMultiScale(gray, 1.3, 5)                             #Detect Faces.
for (x,y,w,h) in faces:                                                 #Loop through all the faces.
    cv2.rectangle(image1,(x,y),(x+w,y+h),(255,0,0),2)                   #Draw a rectangle around them.
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image1[y:y+h, x:x+w]
    eyes = eye.detectMultiScale(roi_gray)                               #Search for eyes in detected faces.
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)      #Draw a rectangle around eyes.
cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
cv2.imshow('Face Detection',image1)                                     #Displays the detected image.
cv2.waitKey(0)
cv2.destroyAllWindows()
