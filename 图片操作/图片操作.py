#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2
import numpy as np
img=cv2.imread("D:/1699194637144535.jpg")


# In[14]:


def CvShow(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[15]:


def ImageZoomOut(img):
    ret=cv2.resize(img,None,fx=0.75,fy=0.75)
    return ret


# In[16]:


def PictureRotation(img):
    h,w=img.shape[:2]
    M=cv2.getRotationMatrix2D((h/2,w/2),135,1)
    ret=cv2.warpAffine(img,M,(h,w))
    return ret


# In[17]:


def PicturePan(img):
    h,w=img.shape[:2]
    M=np.float32([[1,0,-0.75*w],[0,1,-0.8*h]])   
    ret=cv2.warpAffine(img,M,(h,w))
    return ret


# In[18]:


CvShow("img1",ImageZoomOut(img))
CvShow("img2",PictureRotation(ImageZoomOut(img)))
CvShow("img3",PicturePan(img))

