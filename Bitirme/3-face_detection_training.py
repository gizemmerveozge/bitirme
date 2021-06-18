import cv2
import os
from datetime import datetime
import numpy as np
from PIL import Image

cam = cv2.VideoCapture(0)         #use webcam and set video width and height
cam.set(3, 640)
cam.set(4, 480)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Define face classifier

face_id = input('\n enter user id end press <return> ==>  ')  # For each person, enter one numeric face id and name
face_name = input('Please enter your name: ')

count = 0

if os.path.exists("dataset/" + face_name):    #File is created for each person and checked for existence
    print("Name is already saved.")

else:
    os.mkdir("dataset/" + face_name)   # mkdir: creates the file
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            count += 1
            time = int(datetime.now().strftime('%H%M%S%f'))

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/" + face_name + '/' + 'User.' + str(face_id) + '.' + str(time) + ".jpg", gray[y:y + h, x:x + w])
            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30:  # Take 30 face sample and stop video
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

    #-------TRAINING STEP--------

    path = 'dataset/' + face_name   # Path for face image database
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    def getImagesAndLabels(path):   # function to get the images and label data
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')         # converting images to a numpy array of 8-bit positive integers
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = face_detector.detectMultiScale(img_numpy)   #find the face areas in the image numpy array and add them to the faces list.

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids

    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    recognizer.write('trainer.yml')   #Save the model into trainer/trainer.yml
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))  # Print the number of faces trained and end program