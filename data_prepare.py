import numpy as np
import os
import json
import random
from gaze_analysis import gazeAnalysis, json2np, search_data
from config import conf
from tqdm import tqdm
import pdb
'''
1、get all the json file and use gaze_analysis.py to get the feature map
2、scale and encode the feature map
3、split data to train/val sets
4、store the train/val sets to two npy file seperately and save the annotations in cooresponding txt files
'''

train_ratio = 0.8
val_ratio = 0.2

def make_data(root,ab_dir,norm_dir,spec_idx=-1):
    #get file
    abnormal_list = search_data(os.path.join(root,ab_dir),'eye.json')
    normal_list = search_data(os.path.join(root, norm_dir),'eye.json')
    #get feautre
    norm_feat_list = get_feature_list(normal_list,spec_idx = spec_idx)
    print('Normal feature extracted!')
    #pdb.set_trace()
    ab_feat_list = get_feature_list(abnormal_list,spec_idx = spec_idx)
    print('Abnormal feature extracted!')
    #pdb.set_trace()
    #data concat
    all_feat_list = ab_feat_list+norm_feat_list#.extend(norm_feat_list)
    label_list = [1 for i in range(len(ab_feat_list))]+[0 for i in range(len(norm_feat_list))]# ab is 1 and norm is 0
    #shuffle and split dataset
    random.seed(666)
    idx_list = [i for i in range(len(label_list))]
    random.shuffle(idx_list)

    feat_num = len(label_list)
    train_point = int(feat_num*train_ratio)
    train_idx = idx_list[:train_point]
    val_idx = idx_list[train_point:]

    all_feat_arr = np.array(all_feat_list)
    #pdb.set_trace()
    label_arr = np.array(label_list)

    train_feat_list = all_feat_arr[train_idx]
    train_label_list = label_arr[train_idx]
    val_feat_list = all_feat_arr[val_idx]
    val_label_list = label_arr[val_idx]
    #store data
    np.save(os.path.join(root,'train_data.npy'),np.array(train_feat_list))
    np.save(os.path.join(root,'val_data.npy'),np.array(val_feat_list))
    with open(os.path.join(root,'train_label.txt'),'w') as f:
        for v in train_label_list:
            f.write(str(v))
    with open(os.path.join(root,'val_label.txt'),'w') as f:
        for v in val_label_list:
            f.write(str(v))
    print('Data making finished!')

def get_feature_list(file_list,spec_idx=-1):
    result = []
    for file_path in tqdm(file_list):
        with open(file_path, 'r') as f:
            file = json.load(f)
        eye_data = file["gazePoints"]
        gaze_points = json2np(eye_data)
        #check whether result been recorded
        result_path = os.path.join(os.path.dirname(file_path), 'result.json')
        if not (os.path.exists(result_path)):
            continue
        with open(result_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        result_times = result_data['times']
        if('Expression Finish' not in result_times.keys() or 'StroopTest_P1 Start' not in result_times.keys()):
            continue
        #extract feature
        extractor = gazeAnalysis(gaze_points, conf.fixation_radius_threshold, conf.fixation_duration_threshold,
                                 conf.saccade_min_velocity, conf.max_saccade_duration, eye_path=file_path)
        feature = extractor.get_feature_map()
        if(len(feature[feature[:, -1] != -1])==0):#有效event数量为0
            print('error gaze points: '+file_path)
            continue
        if (len(feature[feature[:, -1] == 0]) == 0):
        #if(len(feature[feature:,-1]==4)==0):#WCST为0
            print("error P1: "+file_path)
            continue
        feature = process_feat(feature,spec_idx=spec_idx)#norm and encode feature
        result.append(feature)
    return result


def make_one_hot(arr, max_idx=-1):
    if(max_idx!=-1):
        max_one = max_idx
    else:
        max_one = np.max(arr)
    return np.arange(max_one+1)==arr[:,None].astype(np.integer)

#def get_spec_exp(exp_idx, feature):

def process_feat(feature, spec_idx = -1):
    '''
    delete feature not in experiments
    norm and encode feature
    norm index:0,14,15,16
    :param original feature:
    :return: array
    '''
    #delete feature not in  exp
    #pdb.set_trace()
    feature = feature[feature[:,-1]!=-1]

    if(spec_idx!=-1):#select specific exp data
        feature = feature[feature[:,-1]==spec_idx]
    #get the middle feature
    #pdb.set_trace()
    mid_feat = feature[:,3:14]
    #get duration
    duration_feat = feature[:,2]-feature[:,1]
    #norm middle feature

    mid_normed_feat = mid_feat/mid_feat.max(axis=0)

    #norm duration feature
    duration_feat = (duration_feat/duration_feat.max(axis=0))[:,np.newaxis]

    #one-hot other feature
    event_cls = make_one_hot(feature[:,0])
    begin_area = make_one_hot(feature[:,14],max_idx=9)
    end_area = make_one_hot(feature[:,15],max_idx=9)
    if(spec_idx==-1):
        exp_idx = make_one_hot(feature[:,16],max_idx=5)
        #pdb.set_trace()
        result = np.hstack((event_cls,duration_feat,mid_normed_feat,begin_area,end_area,exp_idx))#(num_event,)
    else:
        result = np.hstack((event_cls,duration_feat,mid_normed_feat,begin_area,end_area))
    #pdb.set_trace()
    return result#(n,40)

if __name__ == '__main__':
    root = 'data/all_data'
    ab_dir = '1'
    norm_dir = '0'
    make_data(root,ab_dir,norm_dir,spec_idx=0)
