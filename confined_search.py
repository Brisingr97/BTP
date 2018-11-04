
# coding: utf-8

# In[2]:


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


# In[10]:


all_folder_paths = ["training/v11","training/v12","training/v21","training/v22"]


# In[3]:


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


# In[33]:


def match_value_between_two_images(matches_k2):
    match_conf = []
    for i in matches_k2:
        match_conf.append(match_confidence(i))
    match_conf.sort()
#     
    ex_val = int(len(match_conf)*0.1)
    return_val = match_conf[:ex_val]
    return (sum(return_val)/ex_val)
    
def total_match(fs1,fs2):
    return match_value_between_two_images(return_matches(fs1,fs2,0))
# def draw_matches(image_1,image_2,matches):
# # def identify_min_set_grayscale(path_of_folder):
#     img3 = cv2.drawMatchesKnn(image_1,kp1,img2,kp2,matches,None,**draw_params)
#     pass


# In[34]:


def h_conf(pre_val,flag):
    d = 1.15
    f = 1.2
    if flag==0:
        return pre_val/d
    else :
        return pre_val*f


# In[35]:


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


# In[12]:


# feature_database.append([])
all_feature_dataset = []
co = 0
for a_folder in all_folder_paths:
    print(a_folder)
    all_feature_dataset.append([])
    one_folder = []
    for subdir, dirs, files in os.walk(a_folder):
        for file in files:
            one_folder.append(file)
    my_files_for_a_folder = sorted(one_folder, key=lambda x: int((x.split('_')[1]).split('.')[0]))
    for i in my_files_for_a_folder:
        file_name = a_folder + '/' + i
        frame = cv2.imread(file_name)
        all_feature_dataset[co].append(reduce_frame_to_features(0,frame))
    co = co+1


# In[42]:


def find_maxima(frame,feature_set):
    max_val = 0  
    for i in range(0,len(feature_set)):
        nn_bha = total_match(frame,feature_set[i])
        if max_val<nn_bha and max_val>0.9*nn_bha:
            continue
        if max_val<nn_bha:
            max_val = nn_bha
        else:
            return i


# In[51]:


# initial path confidence
path_confidence = [0.25,0.25,0.25]

#initial time hypothesis
t_hypothesis = [0,0,0]
t_hypothesis[0] = find_maxima(all_feature_dataset[0][0],all_feature_dataset[1])
t_hypothesis[1] = find_maxima(all_feature_dataset[0][0],all_feature_dataset[2])
t_hypothesis[2] = find_maxima(all_feature_dataset[0][0],all_feature_dataset[3])


for i in range(1,len(all_feature_dataset[0])):
    print(path_confidence)
    print(t_hypothesis)
    this_ob = all_feature_dataset[0][i]
    for k in range(0,len(t_hypothesis)):
        checker = find_maxima(this_ob,all_feature_dataset[k+1])
        print(k+1 , checker)
        if(checker<=t_hypothesis[k]+i+3 and checker >=t_hypothesis[k] ):
            print(k+1,'increasing')
            path_confidence[k] =  h_conf(path_confidence[k],1)
        else:
            print(k+1,'decreasing')
            path_confidence[k] = h_conf(path_confidence[k],0)
#     first image stream
    
    


# In[31]:




