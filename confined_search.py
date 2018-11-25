
# coding: utf-8

# In[85]:


import cv2
import numpy as np
import sys
import os
import math
import time


# In[3]:


def reduce_frame_to_features(frame_object):
    feature_method= cv2.xfeatures2d.SURF_create(400)
    kp,des = feature_method.detectAndCompute(frame_object,None)
    return (kp,des)


# In[37]:


def import_folder_of_images(path_folder):
    images_as_feature_sets = []
    for file_name in os.listdir(path_folder):
        frame = cv2.imread(path_folder+'/'+file_name)
        temp = reduce_frame_to_features(frame)
        if(len(temp[0])>2):
            images_as_feature_sets.append(temp)
    return images_as_feature_sets


# In[32]:


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
    if(len(feature_set1[0])>=2 and len(feature_set2[0])>=2):
        matches = flann.knnMatch(feature_set1[1] ,feature_set2[1],k=2)
    # matchesMask = [[0,0] for i in range(len(matches))]
    else:
        print ("Well, EFF.")
        print(len(feature_set1[0]),len(feature_set1[1]))
        print("---------------")
        print(len(feature_set2[0]),len(feature_set2[1]))
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


# In[26]:


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


# In[27]:


def set_selection(fs):
    my_list_of_features = []
    new_addition = -1
    while(new_addition!=(len(fs)-1)):
        last_unmatched_fs = new_addition+1
#         print(last_unmatched_fs)
        new_addition = last_unmatched_fs+bin_search(fs[last_unmatched_fs:])
#         print("="+str(new_addition))
        my_list_of_features.append(new_addition)
    return my_list_of_features


# In[38]:


all_folders = ["training/p11","training/p12","training/p21","training/p22","training/p31","training/p32","training/p41","training/p42"]


# In[39]:


unpruned_feature_set = []
pruned_feature_set = []
for i in all_folders:
    print("SURFing from - " + i)
    temp_1 = import_folder_of_images(i)
    unpruned_feature_set.append(temp_1)
    print("selecting min-set from fs "+i)
    temp_2 = set_selection(temp_1)
    temp_3 = []
    print("Done, Pruning to new list from- "+str(len(temp_1)))
    for k in temp_2:
        temp_3.append(temp_1[k])
    pruned_feature_set.append(temp_3)
    print("pruned - "+str(len(temp_2)))


# In[44]:


# start hypothesis check


# In[45]:


test_paths = []
test_paths.append(import_folder_of_images("training/k21"))
test_paths.append(import_folder_of_images('training/s31'))


# In[48]:


temp_1 = test_paths[0]


# In[68]:


def match_flagger(frame_fs,test_path):
    limiter = 65
    for i in range(0,len(test_path)):
        if(total_match(frame_fs,test_path[i])>=52):
            return i
            


# In[88]:


start_time = time.time()
for sarthak in range(0,len(unpruned_feature_set)):
    print(sarthak)
    checker = [1]
    time_hypothesis = [0]
    # for unpruned_feature_set[0]
    to_check_path = unpruned_feature_set[sarthak]
    flag = 0
    # past_match = 0
    for i in temp_1:
        if(len(to_check_path)<2 or checker[-1]>=256):
            print("----DONE----")
            print(checker)
            print(time_hypothesis)
            print('\n---------')
            break
        if(checker[-1]<1):
            checker.append(1)
            current_match = match_flagger(i,to_check_path)
            time_hypothesis.append(len(unpruned_feature_set[sarthak])-len(to_check_path))
            continue;
        current_match = match_flagger(i,to_check_path)
        if(flag ==1):
            if(current_match==None):
                print("----DONE----")
                print(checker)
                print(time_hypothesis)
                break
            if(current_match<=2):
                checker.append(checker[-1]*2)
            else:
                checker.append(checker[-1]/4)
        else:
            time_hypothesis.append(current_match)
        to_check_path = to_check_path[current_match:]
#         print(current_match)
#         print(checker)
        flag = 1
print(time.time()-start_time)


# In[87]:


start_time = time.time()
for sarthak in range(0,len(unpruned_feature_set)):
    print(sarthak)
    checker = [1]
    time_hypothesis = [0]
    # for unpruned_feature_set[0]
    to_check_path = pruned_feature_set[sarthak]
    flag = 0
    # past_match = 0
    for i in temp_1:
        if(len(to_check_path)<2 or checker[-1]>=256):
            print("----DONE----")
            print(checker)
            print(time_hypothesis)
            print('\n---------')
            break
        if(checker[-1]<1):
            checker.append(1)
            current_match = match_flagger(i,to_check_path)
            time_hypothesis.append(len(pruned_feature_set[sarthak])-len(to_check_path))
            continue;
        current_match = match_flagger(i,to_check_path)
        if(flag ==1):
            if(current_match==None):
                print("----DONE----")
                print(checker)
                print(time_hypothesis)
                break
            if(current_match<=2):
                checker.append(checker[-1]*2)
            else:
                checker.append(checker[-1]/4)
        else:
            time_hypothesis.append(current_match)
        to_check_path = to_check_path[current_match:]
#         print(current_match)
#         print(checker)
        flag = 1
print(time.time()-start_time)


# In[90]:


71/7.76

