# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 20:14:55 2021

@author: keskiny
"""

# import the opencv library
import cv2
from matplotlib import pyplot as plt
  
  
# define a video capture object
vc = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vc.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    #cv2.imshow('frame', frame)
    height, width, channels = frame.shape
    #print(channels)
    
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vc.release()
# Destroy all the windows
cv2.destroyAllWindows()