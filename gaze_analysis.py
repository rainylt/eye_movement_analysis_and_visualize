#!/usr/bin/python
import numpy as np
import sys, os
import event_detection as ed
from config import names as gs
from config import conf
import json
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
		rad = 0
		for sac in self.saccades:
			vel += sac[9]
			rad += sac[4]
		mean_vel = vel/len(self.saccades)
		mean_rad = rad/len(self.saccades)
		stat = [mean_duration, gaze_freq, mean_vel, mean_rad, len(self.saccades)]

		return stat

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

