#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
from PIL import Image
import numpy as np


# In[2]:


def getImageAndLabels(path):
    facesSamples=[]#储存人脸数据（这是一个二维数组）
    ids=[]#储存姓名数据
    imagePaths=[os.path.join(path,f)for f in os.listdir(path)]  #储存图片信息
    face_detector = cv2.CascadeClassifier("C:/Users/33086/Downloads/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml")
    for imagePath in imagePaths:#遍历列表中的图片
        PIL_img=Image.open(imagePath).convert('L')#以灰度图像的方式打开
        img_numpy=np.array(PIL_img,'uint8')#将图像转化为数组
        faces=face_detector.detectMultiScale(img_numpy)#使用分类器获取人脸特征
        id =int(os.path.split(imagePath)[1].split('.')[0])#提取图片.之前的东西作为id
        for x,y,w,h in faces:
            ids.append(id)#将id添加到ids这个列表内
            facesSamples.append(img_numpy[y:y+h,x:x+w])#把x,y,w,h添加到facesSamples这个列表
    return facesSamples,ids  #返回                          


# In[3]:


if __name__ == '__main__':
    path="D:/positive"
    faces,ids=getImageAndLabels(path)#获取图像数组和id标签数组和姓名
    recognizer=cv2.face.LBPHFaceRecognizer_create()#加载识别器
    recognizer.train(faces,np.array(ids))#训练
    recognizer.write("D:/trainer/trainer.yml")#保存文件

