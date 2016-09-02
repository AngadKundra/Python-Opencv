import cv2
camera = cv2.VideoCapture(0)              #Creates an object to read Video from camera.
while(True):                              
	ret, frame = camera.read()              #Returns a frame and true(non zero) value.
	cv2.imshow('Camera Feed', frame)        #Displays the image.
	if cv2.waitKey(1) & 0xFF == ord('k'):   #Closes Video Capture window by exiting the loop.
		break
camera.release()                          #Important to release to make IO device available.
cv2.destroyAllWindows()
