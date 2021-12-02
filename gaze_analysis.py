#!/usr/bin/python
import numpy as np
import pandas as pd
import sys, os
import event_detection as ed
from config import names as gs
from config import conf
import json
import matplotlib.pyplot as plt
import pdb


class gazeAnalysis (object):
	def __init__(self, gaze, fixation_radius_threshold, fixation_duration_threshold, saccade_min_velocity,max_saccade_duration,
					event_strings=None, ti=0, xi=1, yi=2):
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

		# detect errors, fixations, saccades and blinks
		self.errors = self.detect_errors()
		self.fixations, self.saccades, self.wordbook_string = \
			ed.detect_all(self.gaze, self.errors, self.ti, self.xi, self.yi,
							event_strings=event_strings, fixation_duration_threshold=fixation_duration_threshold,
							fixation_radius_threshold=fixation_radius_threshold, saccade_min_velocity=saccade_min_velocity,
							max_saccade_duration=max_saccade_duration)

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
		plt.plot(x_timeline, y_vel, 'o-',label='vel')
		plt.plot(x_timeline, y_max_angle,'o-', label = 'max_angle')
		plt.plot(x_timeline, y_mean_angle, 'o-',label = 'mean_angle')
		plt.title('saccade event indicator analysis')
		plt.xlabel('event index')
		#plt.ylabel('')
		plt.savefig('output/saccade_analysis.png')

	def reconstruct_feature(self):
		self.rec_fixations = []
		self.rec_saccades = []
		for fix in self.fixations:
			feature_vec = []
			feature_vec.append(0)#fixation cls label
			feature_vec.append(fix[4])#start time
			feature_vec.append(fix[5])#end time
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

			area_start_idx, area_end_idx = self.get_area_idx(feature_vec[3:7])#start coord, end coord
			exp_idx = self.get_exp_idx(feature_vec[1:3])#start time, end time

			feature_vec.append(area_start_idx)
			feature_vec.append(area_end_idx)
			feature_vec.append(exp_idx)
			self.rec_fixations.append(feature_vec)

		for sac in self.saccades:
			feature_vec = []
			feature_vec.append(1)
			feature_list = [5,6,0,1,2,3,13,14,15,16,4,9,10]
			for idx in feature_list:
				feature_vec.append(sac[idx])
			area_start_idx, area_end_idx = self.get_area_idx(feature_vec[3:7])
			exp_idx = self.get_exp_idx(feature_vec[1:3])
			feature_vec.append(area_start_idx)
			feature_vec.append(area_end_idx)
			feature_vec.append(exp_idx)
			self.rec_saccades.append(feature_vec)

	def get_feature_map(self):
		'''
		merge fixations and saccades to a np array
		:return:
		'''

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
		for file in files:
			if(file==filename):
				file_path = os.path.join(root, file)
				result.append(file_path)
	return result

def get_dir_stat(path, filename):
	'''
	get statistic information from all the eye.json in database
	:param path: database path
	:param filename: eye.json
	:return: all features, [num_person, stat_dim]
	'''
	file_list = search_data(path, filename)
	stat_array = np.array([])
	for file_path in file_list:
		with open(file, 'r') as f:
			file = json.load(f)
		eye_data = file["gazePoints"]
		gaze_points = json2np(eye_data)
		extractor = gazeAnalysis(gaze_points, conf.fixation_radius_threshold, conf.fixation_duration_threshold,
								 conf.saccade_min_velocity, conf.max_saccade_duration)
		feature = extractor.stat_analysis()
		stat_array.extend(feature)
	return stat_array

def analyze_avg_feature(stat_array):
	'''
	indicator:
	num_person

	'''
	df = pd.DataFrame(stat_array, columns=['gaze_duration','num_gaze','sac_peak_vel','sac_max_angle','mean_angle','num_sac'])
	df.describe()

if __name__ == '__main__':
	with open('data/JsonData/eye_576.json', 'r') as f:
		file = json.load(f)
	eye_data = file["gazePoints"]
	gaze_points = json2np(eye_data)
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
	#gshape = gaze_points.shape
	extractor = gazeAnalysis(gaze_points, conf.fixation_radius_threshold, conf.fixation_duration_threshold,
								conf.saccade_min_velocity, conf.max_saccade_duration)
	extractor.analyze_fixations()

