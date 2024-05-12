#!/usr/bin/env python
# coding: utf-8

# In[25]:


import cv2
img=cv2.imread("D:/169919463714453.jpg")


# In[27]:


def CvShow(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[ ]:


def FeatureExtraction(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift=cv2.xfeatures2d.SIFT_create()
    kp,des=sift.detectAndCompute(gray,None)
    cv2.drawKeypoints(img,kp,img,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img


# In[28]:


CvShow("img",FeatureExtraction(img))

