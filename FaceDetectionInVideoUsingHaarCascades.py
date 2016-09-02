#Face Detection in video.
import cv2
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')      #Create classifier object for face.
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')                       #Create classifier object for eyes.
camera = cv2.VideoCapture(0)                                                    #Capture Input Video Device.
ret,frame = camera.read()
font = cv2.FONT_HERSHEY_SIMPLEX                                                 #Font type.
while(1):
 ret,frame = camera.read()
 gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),           
 flags=cv2.CASCADE_SCALE_IMAGE)                                                 #Detect Faces.
 for (x, y, w, h) in faces:
    cv2.putText(frame,'Face',(x,y-5), font, .6,(0,255,0),2,cv2.LINE_AA)         #Puts Text on Rectangle.
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)                    #Draw Rectangle on Faces.
    roi_gray = gray[y:y+h,x:x+w]
    roi_color = frame[y:y+h,x:x+w]
    eyes = eyeCascade.detectMultiScale(roi_gray,scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE)                                              #Detect Eyes.
    for (ex,ey,ew,eh) in eyes:
	cv2.putText(roi_color,'eyes',(ex,ey-5), font, .6,(0,255,0),2,cv2.LINE_AA) #Puts text on rectangle.   
    	cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)        #Draws Rectangle on Eyes.
		 
 cv2.imshow("Detecting",frame)                                                  #Display Image.
 k = cv2.waitKey(60) & 0xff                                                     #Exits on Esc.
 if k == 27:
       break
cv2.destroyAllWindows()
camera.release()
