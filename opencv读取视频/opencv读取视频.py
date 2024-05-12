#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
vc=cv2.VideoCapture("D:/Screenrecorder-2023-11-08-21-14-50-465.mp4")


# In[8]:


if vc.isOpened():
    open
else:
    open=False


# In[9]:


while open:
    ret,frame=vc.read()
    if frame is None:
        break
    if ret == True:
        img=frame[120:360,200:600]
        if cv2.waitKey(25) == 113:
            break
        if cv2.waitKey(25) == 32:
            cv2.waitKey(0)
        cv2.imshow('result', img)
vc.release()
cv2.destroyAllWindows()

