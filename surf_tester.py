
# coding: utf-8

# In[1]:


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


# In[42]:


def match_confidence(a_match_object):
    return ((a_match_object[1].distance/a_match_object[0].distance)-1)*100

def return_matches(feature_set1,feature_set2,mode_of_operation):
    # find some heuristic to understand a good match or not
    # maybe ratio test
    # maybe distance FLANN
    FLANN_INDEX_KDTREE = 0
    if mode_of_operation==2:
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    else:
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,dict())
    matches = flann.knnMatch(feature_set1[1] ,feature_set2[1],k=2)
    # matchesMask = [[0,0] for i in range(len(matches))]
    return matches


# In[14]:


def match_value_between_two_images(matches_k2):
    match_conf = []
    for i in matches_k2:
        match_conf.append(match_confidence(i))
    match_conf.sort()
#     
    ex_val = int(len(match_conf)*0.1)
    return_val = match_conf[:ex_val]
    return (sum(return_val)/ex_val)
    

# def draw_matches(image_1,image_2,matches):
# # def identify_min_set_grayscale(path_of_folder):
#     img3 = cv2.drawMatchesKnn(image_1,kp1,img2,kp2,matches,None,**draw_params)
#     pass


# In[78]:


def h_conf(pre_val,flag):
    d = 3
    f = 2
    if flag==0:
        return pre_val/d
    else :
        return pre_val*f


# In[20]:


def reduce_frame_to_features(mode_of_use,frame_object):
    # surf,sift or orb?
    if mode_of_use == None or mode_of_use==0:
        feature_method= cv2.xfeatures2d.SURF_create(400)
    elif mode_of_use==1:
        feature_method = cv2.xfeatures2d.SIFT_create()
    elif mode_of_use ==2:
        feature_method = cv2.ORB()

    kp,des = feature_method.detectAndCompute(frame_object,None)
    return (kp,des)


# In[68]:


a = [cv2.imread('training/1.jpg'),cv2.imread('training/2.jpg'),cv2.imread('training/frame5.jpg'),cv2.imread('training/frame115.jpg')]
b = [cv2.imread('training/3.jpg'),cv2.imread('training/4.jpg'),cv2.imread('training/frame6.jpg'),cv2.imread('training/frame116.jpg')]


# In[75]:


c = 0.25


# In[76]:


for i in range(0,len(a)):
    if (c<0.1) :
        print("NO MATCH")
        break;        
    if(match_value_between_two_images(return_matches(reduce_frame_to_features(0,a[i]),reduce_frame_to_features(0,b[i]),0)))>0.5:
       c =  h_conf(c,1)
    else:
        c = h_conf(c,0)
    if (c>2):
        print("Perfect Match")
        break;
        
    


# In[77]:


c

