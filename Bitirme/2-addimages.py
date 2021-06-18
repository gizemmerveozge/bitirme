import cv2
import os
from datetime import datetime
import numpy as np
from PIL import Image

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n enter user id end press <return> ==>  ') # For each person, enter one numeric face id and name
print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
face_name = input('Please enter your name: ')

count = 0

if os.path.exists("dataset/" + face_name):

    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            count += 1
            time = int(datetime.now().strftime('%H%M%S%f'))
            cv2.imwrite("dataset/" + face_name + '/' + 'User.' + str(face_id) + '.'  + str(time) + ".jpg", gray[y:y + h, x:x + w])
            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30:  # Take 30 face sample and stop video
            break

    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

    path = 'dataset/' + face_name      # Path for face image database
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    def getImagesAndLabels(path):      # function to get the images and label data
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = face_detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        return faceSamples,ids
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer.yml')     # Save the model into trainer/trainer.yml
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))