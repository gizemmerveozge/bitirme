import cv2
import os
import numpy as np
import datetime
from PIL import Image, ImageDraw, ImageFont


def print_utf8_text(image, xy, text, color):  # utf-8 characters
    fontName = 'FreeSerif.ttf'  # 'FreeSansBold.ttf' # 'FreeMono.ttf' 'FreeSerifBold.ttf'
    font = ImageFont.truetype(fontName, 24)  # select font
    img_pil = Image.fromarray(image)  # convert image to pillow mode
    draw = ImageDraw.Draw(img_pil)  # prepare image
    draw.text((xy[0], xy[1]), text, font=font,
              fill=(color[0], color[1], color[2], 0))  # b,g,r,a
    image = np.array(img_pil)  # convert image to cv2 mode (numpy.array())
    return image


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['None']
days = []

tarama = os.scandir("dataset")    #Scan Dataset folder and add names to the array.(names[] array)
for name_ in tarama:
    names.append(name_.name)

cam = cv2.VideoCapture(0)   # Initialize and start realtime video capture
cam.set(3, 1000)
cam.set(4, 800)

minW = 0.1 * cam.get(3)   # Define min window size to be recognized as a face
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id,confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less them 60 ==> "0" is perfect match
        if (confidence<60):
            id = names[id]
            confidence = "  {0}%".format(round(100-confidence))

        else:
            id = "bilinmiyor"
            confidence = "  {0}%".format(round(100-confidence))

        color = (255, 255, 255)
        img = print_utf8_text(img, (x + 5, y - 25), str(id), color)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff
    if k == 27 or k == ord('q'):
        break

time = datetime.date.today()          #adding students names and time to the list
with open("record.txt", "a+") as f:
      f.write(id + "\t" +str(time)  + "\n")


print("\n [INFO] Programdan çıkıyor ve ortalığı temizliyorum")
cam.release()
cv2.destroyAllWindows()