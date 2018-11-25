import cv2
import numpy as np
import sys
import argparse
import os

def extractImages(pathIn, pathOut):
    try:
            
        vidcap = cv2.VideoCapture(pathIn,0)
        success,image = vidcap.read()
        count = 0
        success = True
        while success:
            success,image = vidcap.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         print ('Read a new frame: ', success)
            gray_re = cv2.resize(gray,(640,360))
            count += 1
            if(count%10==0):
                cv2.imwrite( pathOut + "\\frame%d.jpg" % (count/10), gray_re)     # save frame as JPEG file
    except:
        pass

def issue_videos(path_names):  
    pathnumber = 0;
    for i in path_names:    
        try:
            os.mkdir("training/folder"+str(pathnumber))
            extractImages(i,("training/folder"+str(pathnumber)))
            pathnumber = pathnumber+1
        except:
            pass

# issue_videos(["training/v11.mp4","training/v12.mp4","training/v21.mp4","training/v22.mp4"])