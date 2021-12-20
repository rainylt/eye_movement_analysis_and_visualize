#!/usr/bin/python
import numpy as np
import pandas as pd
import sys, os
import event_detection as ed
from config import names as gs
from config import conf
import json
import matplotlib.pyplot as plt

from datetime import datetime
import pdb
from tqdm import tqdm
import re


class gazeAnalysis(object):
	def __init__(self, gaze, fixation_radius_threshold, fixation_duration_threshold, saccade_min_velocity,max_saccade_duration,eye_path,
					event_strings=None, ti=0, xi=1, yi=2, ):
		assert gaze.size > 0

		# save data in instance
		self.gaze = gaze
		self.event_strings = event_strings

		# save constants, indices and thresholds that will be used muttiple times
		self.fixation_radius_threshold = fixation_radius_threshold
		self.fixation_duration_threshold = fixation_duration_threshold
		self.xi = xi
		self.yi = yi
		self.ti = ti

		self.eye_path = eye_path
		self.result_path = os.path.join(os.path.dirname(eye_path),'result.json')
		with open(self.eye_path, 'r') as f:
			eye_data = json.load(f)
		try:
			self.base_time = eye_data['eyeStartTime'].split(' ')[1]
		except:#early formats
			self.base_time = ret = re.search('T(.*?)\.',eye_data['eyeStartTime']).group()[1:-1]
		with open(self.result_path, 'r',encoding='utf-8') as f:
			result = json.load(f)
		self.result_times = result['times']
		# detect errors, fixations, saccades and blinks
		self.errors = self.detect_errors()
		self.fixations, self.saccades, self.wordbook_string = \
			ed.detect_all(self.gaze, self.errors, self.ti, self.xi, self.yi,
							event_strings=event_strings, fixation_duration_threshold=fixation_duration_threshold,
							fixation_radius_threshold=fixation_radius_threshold, saccade_min_velocity=saccade_min_velocity,
							max_saccade_duration=max_saccade_duration)


		#print('event extracting finished!')

	def detect_errors(self, outlier_threshold=0.5):
		"""
		:param outlier_threshold: threshold beyond which gaze must not be outside the calibration area (i.e. [0,1])
		"""
		errors = np.full((len(self.gaze)), False, dtype=bool)

		# gaze is nan
		errors[np.isnan(self.gaze[:, self.xi])] = True
		errors[np.isnan(self.gaze[:, self.yi])] = True

		# gaze outside a certain range
		errors[self.gaze[:, self.xi] < -outlier_threshold] = True
		errors[self.gaze[:, self.xi] > outlier_threshold + 1] = True

		errors[self.gaze[:, self.yi] < -outlier_threshold] = True
		errors[self.gaze[:, self.yi] > outlier_threshold + 1] = True

		return errors

	def get_gaze(self):
		return self.fixations

	def get_sac(self):
		return self.saccades

	def stat_analysis(self):
		#TODO statistically analyze the event information for one user
		'''
		gaze: duration, frequency
		saccade: peak vel, frequency, max angle, mean angle
		:return:
		'''
		duration = []
		for gaze in self.fixations:
			duration.append(gaze[7] - gaze[6])
		mean_duration = np.mean(duration)/30 #frames/30 = time
		gaze_freq = len(self.fixations)
		vel = 0
		max_a = 0
		mean_a = 0
		for sac in self.saccades:
			vel += sac[9]
			max_a += sac[11]
			mean_a += sac[12]
		mean_vel = vel/len(self.saccades)
		mean_max_angle = max_a/len(self.saccades)
		mean_angle = mean_a/len(self.saccades)
		stat = [mean_duration, gaze_freq, mean_vel, mean_max_angle, mean_angle, len(self.saccades)]

		return stat

	def time_series_analysis(self):
		'''
		draw the curve of the indicators
		扫视的峰值速度、最大角度、平均角度的变化[9,11,12]
		:return:
		'''
		x_timeline = [x for x in range(len(self.saccades))]
		y_vel = [x[9] for x in self.saccades]
		y_max_angle = [x[11] for x in self.saccades]
		y_mean_angle = [x[12] for x in self.saccades]
		#plt.plot(x_timeline, y_vel, '-',label='vel')
		plt.plot(x_timeline, y_max_angle,'-', label = 'max_angle')
		#plt.plot(x_timeline, y_mean_angle, 'o',label = 'mean_angle')
		plt.title('saccade event indicator analysis')
		plt.xlabel('event index')
		plt.legend()
		#plt.ylabel('')
		plt.savefig('output/saccade_analysis.png')

	def reconstruct_feature(self):
		'''
		0 类别
		1 开始时间
		2 结束时间（是否用duration更好？）
		#1 duration
		3-4 开始坐标
		5-6 结束坐标
		7-8 坐标均值
		9-10 坐标方差
		11 radians
		12 峰值速度
		13 振幅
		14 开始区域
		15 结束区域
		16 实验序号
		:return:
		'''
		self.rec_fixations = []
		self.rec_saccades = []
		for fix in self.fixations:
			feature_vec = []
			feature_vec.append(0)#fixation cls label (need one hot)
			feature_vec.append(fix[4])#start time
			feature_vec.append(fix[5])#end time
			#feature_vec.append(fix[5]-fix[4])#duration
			feature_vec.append(fix[10])
			feature_vec.append(fix[11])
			feature_vec.append(fix[12])
			feature_vec.append(fix[13])#start coord and end coord
			feature_vec.append(fix[0])
			feature_vec.append(fix[1])#mean coord
			feature_vec.append(fix[2])
			feature_vec.append(fix[3])#var coord
			feature_vec.append(fix[14])#radians
			feature_vec.append(fix[15])#peak vel
			feature_vec.append(fix[16])#amplitude

			exp_idx = self.get_exp_idx(feature_vec[1:3])  # start time, end time
			if(exp_idx != -1):
				area_start_idx, area_end_idx = self.get_area_idx(feature_vec[3:7], exp_idx)#start coord, end coord
			else:
				area_start_idx, area_end_idx = -1, -1


			feature_vec.append(area_start_idx)#(need one hot)
			feature_vec.append(area_end_idx)#(need one hot)
			feature_vec.append(exp_idx)#(need one hot)
			self.rec_fixations.append(feature_vec)
		#print('fixitions reconstructing finished!')
		for sac in self.saccades:
			feature_vec = []
			feature_vec.append(1)
			#feature_vec.append(sac[5])
			#feature_vec.append(sac[6])
			#feature_vec.append(sac[6]-sac[5])
			feature_list = [5,6,0,1,2,3,13,14,15,16,4,9,10]
			for idx in feature_list:
				feature_vec.append(sac[idx])

			exp_idx = self.get_exp_idx(feature_vec[1:3])
			if (exp_idx != -1):
				area_start_idx, area_end_idx = self.get_area_idx(feature_vec[3:7],exp_idx)
			else:
				area_start_idx, area_end_idx = -1, -1
			feature_vec.append(area_start_idx)
			feature_vec.append(area_end_idx)
			feature_vec.append(exp_idx)
			self.rec_saccades.append(feature_vec)
		#print(len(self.rec_fixations[0]))
		#print(len(self.rec_saccades[0]))
		#print('saccades reconstructing finished!')


	def get_exp_idx(self, time_vec):
		'''
		根据result.json找到各阶段时间
		根据event时间判断在哪个experiment
		:param time_vec:
		:return:
		'''
		def get_rel_time(time_str,time_str_base):
			base_time = datetime.strptime(time_str_base,'%H:%M:%S')
			now_time = datetime.strptime(time_str,'%H:%M:%S')
			dur = (now_time-base_time).seconds
			#pdb.set_trace()
			return dur#(datetime.strptime(time_str,'%H:%M:%S')-datetime.strptime(time_str_base,'%H:%M:%S')).seconds
		base_time = self.base_time
		times = self.result_times
		#if('StroopTest_P1 Start' not in times.keys()):
			#print('Key Error!')
			#print(self.eye_path)

		stroop1_start = get_rel_time(times['StroopTest_P1 Start'].split(' ')[1], base_time)
		#stroop1_end = get_rel_time(times['StroopTest_P1 Finish'].split(' ')[1], base_time)
		stroop2_start = get_rel_time(times['StroopTest_P2 Start'].split(' ')[1], base_time)
		#stroop2_end = get_rel_time(times['StroopTest_P2 Finish'].split(' ')[1], base_time)
		stroop3_start = get_rel_time(times['StroopTest_P3 Start'].split(' ')[1], base_time)
		#stroop3_end = get_rel_time(times['StroopTest_P3 Finish'].split(' ')[1], base_time)
		stroop4_start = get_rel_time(times['StroopTest_P4 Start'].split(' ')[1], base_time)
		#stroop4_end = get_rel_time(times['StroopTest_P4 Finish'].split(' ')[1], base_time)

		wcst_start = get_rel_time(times['WCST Start'].split(' ')[1], base_time)
		#wcst_end = get_rel_time(times['WCST Finish'].split(' ')[1], base_time)

		expression_start = get_rel_time(times['Expression Start'].split(' ')[1], base_time)
		#if('Expression Finish' not in times.keys()):
			#print(self.eye_path)
		expression_end = get_rel_time(times['Expression Finish'].split(' ')[1], base_time)
		time_list = [stroop1_start,stroop2_start,stroop3_start,stroop4_start,wcst_start,expression_start,expression_end]
		#pdb.set_trace()
		for i in range(len(time_list)-1):#i~[0,5]
			if(time_vec[0]>time_list[i] and time_vec[1]<time_list[i+1]):
				return i
		return -1#不在范围内id就为-1
	def get_area_idx(self, coord_vec, exp_idx):
		def search_bbox(bbox_list, coord_vec):
			start_area = 0# background
			end_area = 0
			for id,bbox in bbox_list.items():
				if(bbox[2]>=coord_vec[0]>=bbox[0] and bbox[3]>=coord_vec[1]>=bbox[1]):
					start_area = id
				if (bbox[2] >= coord_vec[2] >= bbox[0] and bbox[3] >= coord_vec[3] >= bbox[1]):
					end_area = id
			return start_area, end_area
		if(exp_idx<=3):
			bbox_list = {1:[192,117,522,440],2:[772,70,992,190],3:[772,208,992,328],4:[772,345,992,466],5:[772,484,992,604]}
			return search_bbox(bbox_list, coord_vec)
		if(exp_idx==4):
			bbox_list = {1:[484,0,616,65],2:[0,66,229,229],3:[229,66,532,229],4:[532,66,837,229],5:[839,66,1102,229],
						 6:[0,326,222,450],7:[370,278,733,549],8:[511,571,593,597],9:[821,377,1010,450]}
			return search_bbox(bbox_list, coord_vec)
		if(exp_idx==5):
			bbox_list = {1: [137,139,698,550], 2: [723,139,885,255], 3: [916,139,1081,255], 4: [723,288,995,402],
						 5: [916,288,1081,402], 6: [723,435,885,551], 7: [916,435,1081,551]}
			return search_bbox(bbox_list, coord_vec)

	def get_feature_map(self):
		'''
		1、translate index feature to one hot feautre
		2、merge fixations and saccades to a np array
		:return: feature map (numpy array)
		'''
		#feature_map = np.array([[]])
		ga_idx = 0
		sac_idx = 0
		self.reconstruct_feature()
		feature_map = np.array([])
		#print("feature reconstructing finished!")
		#pdb.set_trace()
		#for ga in self.rec_fixations:
			#if(len(ga)!=17):
				#print(len(ga))
		#pdb.set_trace()
		while(ga_idx<len(self.rec_fixations) and sac_idx<len(self.rec_saccades)):
			if(self.rec_fixations[ga_idx][1]<self.rec_saccades[sac_idx][1]):
				if(feature_map.size == 0):
					feature_map = np.array(self.rec_fixations[ga_idx])
				else:
					feature_map = np.row_stack((feature_map,self.rec_fixations[ga_idx]))
				#print("error ga idx:"+str(ga_idx))
				ga_idx += 1
			else:
				if (feature_map.size == 0):
					feature_map = np.array(self.rec_saccades[sac_idx])
				else:
					feature_map = np.row_stack((feature_map,self.rec_saccades[sac_idx]))
				#print("error sac idx:"+str(sac_idx))
				sac_idx += 1
		#print
		#pdb.set_trace()
		if(ga_idx==len(self.rec_fixations)):
			while(sac_idx<len(self.rec_saccades)):
				feature_map = np.row_stack((feature_map,self.rec_saccades[sac_idx]))
				sac_idx += 1
		else:
			while(ga_idx<len(self.rec_fixations)):
				feature_map = np.row_stack((feature_map,self.rec_fixations[ga_idx]))
				ga_idx += 1
		return feature_map

def json2np(eye_data):
	event_list = []
	frame_idx = 0
	data_idx = 0
	while (data_idx < len(eye_data)):
		guide_time = round(frame_idx * 0.033, 6)
		data = eye_data[data_idx]
		act_time = round(data["timestamp_us"] * 10 ** (-6), 6)
		if (guide_time + 0.01 < act_time):
			event_list.append([guide_time, np.nan, np.nan])
		else:
			event_list.append([act_time, data["position"]["x"], data["position"]["y"]])
			data_idx += 1
		frame_idx += 1
	gaze_points = np.array(event_list)
	return gaze_points

def search_data(path, filename):
	'''
	search file in path
	:param path: root path storing files
	:param filename: the name to be searched
	:return: searched file path list
	'''
	result = []
	for root, dirs, files in os.walk(path):
		#pdb.set_trace()
		for file in files:
			if(file==filename):
				file_path = os.path.join(root, file)
				result.append(file_path)
	return result

def get_dir_stat(path, filename,sheet):
	'''
	get statistic information from all the eye.json in database
	:param path: database path
	:param filename: eye.json
	:return: all features, [num_person, stat_dim]
	'''
	file_list = search_data(path, filename)
	#pdb.set_trace()
	stat_array = np.array([[]])
	flag = 0
	#read excel file
	user_info = pd.read_excel('data/info.xlsx', sheet_name=sheet)
	#pdb.set_trace()
	for file_path in tqdm(file_list):
		with open(file_path, 'r') as f:
			file = json.load(f)
		result_path = os.path.join(os.path.dirname(file_path), 'result.json')
		if not (os.path.exists(result_path)):
			continue
		with open(result_path, 'r', encoding='utf-8') as f:
			result_data = json.load(f)
		result_times = result_data['times']
		if ('Expression Finish' not in result_times.keys() or 'StroopTest_P1 Start' not in result_times.keys()):
			continue
		#get the user name
		user = re.findall('\\\\(.*?)_',file_path)[0]
		#get the sex and grade
		user_line = user_info[user_info['姓名']==user]
		if(user_line.empty):#user not found
			continue
		sex_dict = {'男':0,'女':1}
		try:
			sex = sex_dict[user_line.iat[0,3]]
		except:
			print(user+' not found error, continue')
			continue
		grade = user_line.iat[0,2]
		user_add = [sex, grade]
		#pdb.set_trace()
		eye_data = file["gazePoints"]
		gaze_points = json2np(eye_data)
		extractor = gazeAnalysis(gaze_points, conf.fixation_radius_threshold, conf.fixation_duration_threshold,
								 conf.saccade_min_velocity, conf.max_saccade_duration,eye_path=file_path)
		feature = extractor.stat_analysis()
		#add user info
		feature = feature+user_add
		#pdb.set_trace()
		if(flag==0):
			stat_array = np.expand_dims(np.array(feature),0)
			flag =1
		else:
			stat_array = np.row_stack((stat_array,feature))
	#pdb.set_trace()
	return stat_array

def analyze_avg_feature(stat_array):
	'''
	indicator:
	num_person

	'''
	df = pd.DataFrame(stat_array, columns=['gaze_duration','num_gaze','sac_peak_vel','sac_max_angle','mean_angle','num_sac','sex','grade'])
	pdb.set_trace()
	df.describe()
	'''
	df.groupby('sex').describe().loc[0]
	df.groupby('sex').describe().loc[1]
	df.groupby('grade').describe().loc[0]
	'''
	df['gaze_duration'].groupby(df['sex']).describe()

if __name__ == '__main__':
	data_path = 'data/all_data/1'
	filename = 'eye.json'
	stat_array = get_dir_stat(data_path, filename,'erbao')
	analyze_avg_feature(stat_array)
	'''
	eye_path = 'data/JsonData/eye_576.json'
	with open(eye_path, 'r') as f:
		file = json.load(f)
	eye_data = file["gazePoints"]
	gaze_points = json2np(eye_data)
	'''
	'''
	test_list = np.array([[0,0.1,0.1]])
	#test_list = []
	for i in range(1,2):
		test_list = np.row_stack((test_list,[i*0.03, 0.1, 0.1]))
	for i in range(2,10):
		test_list = np.row_stack((test_list,[i*0.03, np.nan, np.nan]))
	test_list = np.row_stack((test_list,[0.3, 0.9, 0.9]))
	test_list = np.row_stack((test_list, [0.3, 0.9, 0.9]))
	test_list = np.row_stack((test_list, [0.3, 0.9, 0.9]))
	gaze_points = test_list
	#import pdb
	#pdb.set_trace()
	#print(gaze_points)
	'''
	'''
	#gshape = gaze_points.shape
	extractor = gazeAnalysis(gaze_points, conf.fixation_radius_threshold, conf.fixation_duration_threshold,
								conf.saccade_min_velocity, conf.max_saccade_duration,eye_path=eye_path)
	#extractor.time_series_analysis()
	#extractor.analyze_fixations()
	feautre_map = extractor.get_feature_map()
	print(feautre_map.shape)
	'''

