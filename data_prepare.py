import numpy as np
import os
import json
import random
from gaze_analysis import gazeAnalysis, json2np, search_data
from config import conf
'''
1、get all the json file and use gaze_analysis.py to get the feature map
2、scale and encode the feature map
3、split data to train/val sets
4、store the train/val sets to two npy file seperately and save the annotations in cooresponding txt files
'''

train_ratio = 0.8
val_ratio = 0.2

def make_data(root,ab_dir,norm_dir):
    #get file
    abnormal_list = search_data(os.path.join(root,ab_dir),'eye.json')
    normal_list = search_data(os.path.join(root, norm_dir),'eye.json')
    #get feautre
    ab_feat_list = get_feature_list(abnormal_list)
    norm_feat_list = get_feature_list(normal_list)
    #data concat
    all_feat_list = ab_feat_list.extend(norm_feat_list)
    label_list = [1 for i in range(len(ab_feat_list))].extend([0 for i in range(len(norm_feat_list))])# ab is 1 and norm is 0
    #shuffle and split dataset
    random.seed(666)
    idx_list = [i for i in range(len(label_list))]
    random.shuffle(idx_list)

    feat_num = len(label_list)
    train_point = feat_num*train_ratio
    train_idx = idx_list[:train_point]
    val_idx = idx_list[train_point:]

    train_feat_list = all_feat_list[train_idx]
    train_label_list = label_list[train_idx]
    val_feat_list = all_feat_list[val_idx]
    val_label_list = label_list[val_idx]
    #store data
    np.save(np.array(train_feat_list),'train_data.npy')
    np.save(np.array(val_feat_list),'val_data.npy')
    with open(os.path.join(root,'train_label.txt')) as f:
        for v in train_label_list:
            f.write(v)
    with open(os.path.join(root,'val_label.txt')) as f:
        for v in val_label_list:
            f.write(v)

def get_feature_list(file_list):
    result = []
    for file_path in file_list:
        with open(file_path, 'r') as f:
            file = json.load(f)
        eye_data = file["gazePoints"]
        gaze_points = json2np(eye_data)
        extractor = gazeAnalysis(gaze_points, conf.fixation_radius_threshold, conf.fixation_duration_threshold,
                                 conf.saccade_min_velocity, conf.max_saccade_duration, eye_path=file_path)
        feature = extractor.get_feature_map()
        feature = process_feat(feature)#norm and encode feature
        result.append(feature)
    return result


def make_one_hot(arr):
    max_one = np.max(arr)
    return np.arange(max_one+1)==arr[:,None].astype(np.integer)
def process_feat(feature):
    '''
    norm and encode feature
    norm index:0,14,15,16
    :param original feature:
    :return: array
    '''
    #get the middle feature
    mid_feat = feature[:,1:14]
    #norm middle feature
    mid_normed_feat = mid_feat/mid_feat.max(axis=0)
    #one-hot other feature
    event_cls = make_one_hot(feature[:,0])
    begin_area = make_one_hot(feature[14])
    end_area = make_one_hot(feature[15])
    exp_idx = make_one_hot(feature[16])
    result = np.hstack(event_cls,mid_normed_feat,begin_area,end_area,exp_idx)
    return result

if __name__ == '__main__':
    root = 'data/all_data'
    ab_dir = '1'
    norm_dir = '0'
    make_data(root,ab_dir,norm_dir)
