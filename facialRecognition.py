import cv2, os
import numpy as np
from PIL import Image

cascadePath = "/home/angad/Desktop/haarcascade_frontalface_default.xml"    #Face Cascade Classifier path.
faceCascade = cv2.CascadeClassifier(cascadePath)                           #Create Face Cascade object.  

recognizer = cv2.face.createLBPHFaceRecognizer()                           #Create LBPH face recognizer object.  

def get_images_and_labels(path):                                           #Using the yale database for Recognition. 
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')] #Detect all faces other than the sad face for training.
    images = []
    labels = []                                                            #Label the faces detected.  
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')                    #Convert to grayscale.
        image = np.array(image_pil, 'uint8')                               #Convert image to numpy array.
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))  #Get the label of the image.
        faces = faceCascade.detectMultiScale(image)                        #Detect the face in the image.           
        for (x, y, w, h) in faces:                                         #If face is detected, append the face to images and the label to labels.
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    #return the images list and labels list
    return images, labels

# Path to the Yale Dataset
path = '/home/angad/Desktop/yalefaces'
# Call the get_images_and_labels function and get the face images and the 
# corresponding labels
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))

# Append the images with the extension .sad into image_paths
image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')]
for image_path in image_paths:
    predict_image_pil = Image.open(image_path).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        if nbr_actual == nbr_predicted:
            print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
        else:
            print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)
        cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
        cv2.waitKey(1000)
