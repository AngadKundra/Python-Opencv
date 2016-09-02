import cv2
image1=cv2.imread('file_path',0)              #for reading in gray scale.
#image1=cv2.imread('file_path',1)             #for reading in RGB scale.
cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE) #for creating a window which automatically adjusts with the size of the image.
#cv2.namedWindow('image', cv2.WINDOW_NORMAL)  #for normal sized window.
cv2.imshow('image',image1)                    #displays the image in the window named 'image'.
cv2.waitKey(0)                                #Waits for a keyboard input before closing any windows.    
cv2.destroyAllWindows()
