import numpy as np#数据
from matplotlib.animation import FuncAnimation#动态图
from matplotlib import pyplot as plt#创建子图
import json#读取文件
import pdb
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

#config
start_time = 752
end_time = 935
start_frame = start_time*30
end_frame = end_time*30
img_width = 1102
img_height = 620
save_frames = end_frame - start_frame

gd_img_path = 'data/gd_img/Expression_1102x620.png'
save_name = 'Expression.mp4'
#gd_img_path = 'data/gd_img/Stroop_1_1102x620.png'
#gd image
stroop1 = plt.imread(gd_img_path)

with open('data/JsonData/eye_576.json', 'r') as f:
    file = json.load(f)
eye_data = file["gazePoints"]
gaze_points = json2np(eye_data)
gaze_points[:,1] = gaze_points[:,1]*img_width
gaze_points[:,2] = gaze_points[:,2]*img_height
#pdb.set_trace()
print("frames length: "+str(gaze_points.shape[0]))
fig, ax = plt.subplots()
fig.set_tight_layout(True)
#plt.xlim(0,1)
#plt.ylim(0,1)
'''
x_data = []
y_data = []

for i in range(1000):
	x_data.append(gaze_points[i][1])
	y_data.append(gaze_points[i][2])
plt.plot(x_data,y_data,'-o')
plt.show()
'''
ax.imshow(stroop1)
sca, = plt.plot(gaze_points[start_frame][1],gaze_points[start_frame][2],'r-o')

def update(i):#i:第i帧，即第i行数据
    x_data = []
    y_data = []
    for m in range(-5,6):
        if((i+m)<0 or (i+m)>=save_frames):
            continue
        x_data.append(gaze_points[i+m][1])
        y_data.append(gaze_points[i+m][2])
    sca.set_data(x_data,y_data)
    return sca
ani = FuncAnimation(fig = fig,
                    func = update,
                    frames = save_frames,
                    interval = 33,
                    blit = False)
print("begin to save!")
ani.save(save_name, writer='ffmpeg', fps=30)
print("save finished")
