#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
img=cv2.imread("D:/111111.jpg")


# In[2]:


def ImageProcessing(img):
    ret1=cv2.resize(img,None,fx=0.75,fy=0.75)
    h,w=ret1.shape[:2]
    M=cv2.getRotationMatrix2D((h/2,w/2),135,1)
    ret2=cv2.warpAffine(ret1,M,(h,w))
    M=np.float32([[1,0,-0.75*w],[0,1,-0.8*h]])   
    ret3=cv2.warpAffine(ret2,M,(h,w))
    return ret3


# In[3]:


def CvShow(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[4]:


def FLANNmatch(img1,img2):
    gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    sift=cv2.xfeatures2d.SIFT_create()
    kp1,des1=sift.detectAndCompute(gray1,None)
    kp2,des2=sift.detectAndCompute(gray2,None)
    FLANN_INDEX_KDTREE=1
    index_params=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    search_params=dict(checks=50)
    flann=cv2.FlannBasedMatcher(index_params,search_params)
    matches=flann.knnMatch(des1,des2,k=2)
    matchesMask=[[0,0] for i in range(len(matches))]
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params=dict(matchColor=(0,255,0),singlePointColor=(0,0,255),matchesMask=matchesMask,flags=0)
    img3=cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    return img3


# In[5]:


CvShow("img",FLANNmatch(ImageProcessing(img),img))

