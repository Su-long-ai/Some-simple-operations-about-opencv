#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2
import os
import numpy as np


# In[14]:


recogizer=cv2.face.LBPHFaceRecognizer_create()#创建一个人脸识别器对象
recogizer.read("D:/trainer/trainer.yml")#加载识别器


# In[15]:


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


# In[16]:


img=cv2.imread("D:/[@SVUV8T3)P]O3[T6)G7_PE.png")
FaceDetection(img)
cv2.waitKey(0)
cv2.destroyAllWindows()

