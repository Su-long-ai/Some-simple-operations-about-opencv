#!/usr/bin/env python
# coding: utf-8

# In[88]:


import cv2
import os
import numpy as np


# In[89]:


recogizer=cv2.face.LBPHFaceRecognizer_create()#创建一个人脸识别器对象
recogizer.read("D:/trainer/trainer.yml")#加载识别器   


# In[90]:


def FaceDetection(img):
    gary = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_detect = cv2.CascadeClassifier("C:/Users/33086/Downloads/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml")
    face = face_detect.detectMultiScale(gary)
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=1)
        cv2.rectangle(img,(x-10,y-10),(x+w+10,y+h+10),color=(0,255,0),thickness=1)
        ids,confidence=recogizer.predict(gary[y:y+h,x:x+w])#confidence为置信评分
        if confidence > 80:
            cv2.putText(img, 'kiki', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    cv2.imshow('result', img)


# In[91]:


vc=cv2.VideoCapture("D:/VID_20231121_192509.mp4")
if vc.isOpened():
    open, frame=vc.read()
else:
    open=False


# In[92]:


while open:
    ret,frame=vc.read()
    if frame is None:
        break
    if ret == True:
        img=frame[60:300,200:600]
        FaceDetection(img)
        if cv2.waitKey(25) == 113:
             break
        if cv2.waitKey(25) == 32:
            cv2.waitKey(0)
vc.release()
cv2.destroyAllWindows()

