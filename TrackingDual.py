import os
import cv2 
from imutils import face_utils
import dlib
import cv2
import numpy as np
from dualModel import DualModel
from torch.optim import Adam
from torch import nn 
from sklearn.model_selection import train_test_split
import numpy as np 
import torch 
import os
import cv2
from time import time 


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

cap = cv2.VideoCapture(0)
# yres = 320
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, yres)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ((yres//3)*4))

model = DualModel().to(device=device)
model.load_state_dict(torch.load("fourClassModelDualModel.pth", map_location=device))
model.eval()

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

def getRect():
    _, image = cap.read()
    while image is None:
        image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    return (rects, gray, image)

def getEyes(rects, gray, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # image = image[shape[38][1]-40:shape[42][1]+40 , shape[37][0]-60:shape[40][0]+60]
        image = image[shape[38][1]-10:shape[42][1]+10 , shape[37][0]-20:shape[40][0]+20] 
        image = cv2.resize(image, (100,50))
        # image = np.array(image)
        cv2.imshow("mata", image)

        top = max([max(x) for x in image])
        image = (torch.from_numpy(np.array([[image]])).to(dtype=torch.float, device=device, non_blocking=True)) / top
    
    return image

def getFace(rects, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for (i,rect) in enumerate(rects):
        (x,y,w,h) = face_utils.rect_to_bb(rect=rect)
        image = image[y:y+h, x:x+w]
        image = cv2.resize(image, (100,100))
        # image = np.array(image)
        cv2.imshow("muka", image)

        top = max([max(x) for x in image])
        image = (torch.from_numpy(np.array([[image]])).to(dtype=torch.float, device=device, non_blocking=True)) / top
    return image

label = ['kiri atas', 'kanan atas', 'kiri bawah', 'kanan bawah']

getTime = []
cnnTime = []
facialLandmarkTime = []
totalTime = []

with torch.no_grad():
    while True:
        timer2 = time()
        (rects, gray, image) = getRect()
        facialLandmarkTime.append(time() - timer2)
        if len(rects) > 0:
            timer = time()
            mata = getEyes(rects, gray, image)
            muka = getFace(rects, image)
            getTime.append(time() - timer)
            output = model(mata, muka)
            prediction = torch.max(output, 1)
            cnnTime.append(time() - timer - getTime[-1])
            print(label[int(prediction.indices)])
            totalTime.append(time() - timer)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

print(f"average getTime = {sum(getTime)/len(getTime)}")
print(f"average cnnTime = {sum(cnnTime)/len(cnnTime)}")
print(f"average facialLandmarkTime = {sum(facialLandmarkTime)/len(facialLandmarkTime)}")
print(f"average FPS = {1/(sum(totalTime)/len(totalTime))}")
            

