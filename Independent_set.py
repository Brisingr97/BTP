
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import sys
import os
import math


# In[2]:


path = "training/p31"


# In[3]:


def reduce_frame_to_features(frame_object):
    feature_method= cv2.xfeatures2d.SURF_create(400)
    kp,des = feature_method.detectAndCompute(frame_object,None)
    return (kp,des)


# In[34]:


def import_folder_of_images(path_folder):
    images_as_feature_sets = []
    for file_name in os.listdir(path_folder):
        frame = cv2.imread(path_folder+'/'+file_name)
        temp = reduce_frame_to_features(frame)
        if(len(temp[0])>2):
            images_as_feature_sets.append(temp)
    return images_as_feature_sets


# In[33]:


def match_confidence(a_match_object):
#         print(a_match_object[0].distance,a_match_object[1].distance)
    return 100 -100*(a_match_object[0].distance/a_match_object[1].distance)

def return_matches(feature_set1,feature_set2,mode_of_operation):
    # find some heuristic to understand a good match or not
    # maybe ratio test
    # maybe distance FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,dict())
    matches = flann.knnMatch(feature_set1[1] ,feature_set2[1],k=2)
    # matchesMask = [[0,0] for i in range(len(matches))]
    return matches
def match_value_between_two_images(matches_k2):
    match_conf = []
    for i in matches_k2:
        match_conf.append(match_confidence(i))
    match_conf.sort()
#     
    ex_val = int(len(match_conf)*0.9)
    return_val = match_conf[ex_val:]
#     print(return_val)
    return (sum(return_val)/len(return_val))
    
def total_match(fs1,fs2):
    return match_value_between_two_images(return_matches(fs1,fs2,0))
# def draw_matches(image_1,image_2,matches):
# # def identify_min_set_grayscale(path_of_folder):
#     img3 = cv2.drawMatchesKnn(image_1,kp1,img2,kp2,matches,None,**draw_params)
#     pass


# In[29]:


def bin_search(fs):
    limiter = 52
    if(len(fs)==1):
        return 0
    last_best_match = 0
    matching_index = 1
    found_match = 0;
    while(found_match==0):
        print('\t',last_best_match,matching_index)
#         if(matching_index>=len(fs)):
#             matching_index = len(fs)-1
        temp = total_match(fs[0],fs[matching_index])
#         if(abs(last_best_match-matching_index)<=1):
#             if(temp<75):
#                 found_match = 1
#                 return last_best_match
#             if(total_match(fs[0],fs[matching_index+1])<75):
#                 found_match = 1
#                 last_best_match = matching_index
#                 break

        
        if(temp>limiter):
            last_best_match = matching_index
            if(matching_index==len(fs)-1):
                return last_best_match
            matching_index = matching_index*2
            if (matching_index>(len(fs)-1)):
                matching_index = len(fs)-1
        elif(temp==limiter):
            last_best_match = matching_index
            found_match = 1
            break
        else:
            matching_index = int((matching_index+last_best_match)/2)
            if(abs(matching_index-last_best_match)<=1):
                if(total_match(fs[0],fs[matching_index])<limiter):
                    break;
                else:
                    return matching_index
                    
    return last_best_match


# In[11]:


def set_selection(fs):
    my_list_of_features = []
    new_addition = -1
    while(new_addition!=(len(fs)-1)):
        last_unmatched_fs = new_addition+1
        print(last_unmatched_fs)
        new_addition = last_unmatched_fs+bin_search(fs[last_unmatched_fs:])
        print("="+str(new_addition))
        my_list_of_features.append(new_addition)
    return my_list_of_features


# In[12]:


p31_features = import_folder_of_images(path)


# In[30]:


abba = set_selection(p31_features)


# In[31]:


len(abba)


# In[18]:


len(p31_features)


# In[28]:


total_match(p31_features[43],p31_features[50])

