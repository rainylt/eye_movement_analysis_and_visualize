import numpy as np#数据
from matplotlib.animation import FuncAnimation#动态图
from matplotlib import pyplot as plt#创建子图
import json#读取文件
from gaze_analysis import gazeAnalysis as ga
from config import conf
from moviepy.editor import VideoFileClip, clips_array
import pdb

class ploter(object):
    '''
    visualize eye-movement data on image
    ***major func:
    draw_gazePoints
    draw_events
    merge_ani
    '''
    def __init__(self, eye_move_path, start_time, end_time, img_width, img_height, gd_path, event_save_path, point_save_path):
        self.start_frame = start_time*30#30Hz
        self.end_frame = end_time*30
        self.save_frames = self.end_frame - self.start_frame
        self.img_width = img_width
        self.img_height = img_height
        self.gd_path = gd_path#底图
        self.event_save_path = event_save_path
        self.point_save_path = point_save_path
        #self.merge_save_path = merge_save_path
        self.eye_move_path = eye_move_path#eye.json

        self.prepare_data()

    def prepare_data(self):
        self.ori_npData, self.npData = self.get_npData()#gaze point的np data;坐标变换后的data
        self.gd_img = self.get_gd()#背景图
        self.ori_gaze_events, self.ori_sac_events = self.get_events()#没有坐标变换的event,已经限制了时间段
        self.gaze_events, self.sac_events = self.eventCoordTrans()#坐标变换
        #pdb.set_trace()
        self.orderedEvents = self.get_orderedEvents()#gaze和sac融合排序

    def draw_gazePoints(self):
        '''
        1、创建画布
        2、贴背景图
        3、设置更新函数
        4、保存视频
        '''
        fig_points, ax_points = plt.subplots()
        ax_points.imshow(self.gd_img)#加底图
        gazePoints = self.npData

        sca, = plt.plot(gazePoints[start_frame][1], gazePoints[start_frame][2], 'r-o')#初始点，创建sca对象
        base_idx = self.start_frame
        def update(i):  # i:第i帧，即第i行数据
            x_data = []
            y_data = []
            abs_idx = base_idx + i
            for m in range(-5, 6):#每次画周围10帧
                if ((i + m) < 0 or (i + m) >= self.save_frames):
                    continue
                x_data.append(gazePoints[abs_idx + m][1])#这里应该是gazePoints的第self.start_frame+i+m帧
                y_data.append(gazePoints[abs_idx + m][2])
            sca.set_data(x_data, y_data)#更新scatter坐标
            return sca

        ani = FuncAnimation(fig=fig_points,#画布
                            func=update,#更新函数
                            frames=self.save_frames,
                            interval=33,#因为保存时单独设置fps，这个参数这里其实无意义
                            blit=False)#共 self.save_frame帧，每帧update一次

        ani.save(self.point_save_path, writer='ffmpeg', fps=30)
        print('Draw points finished！')

    def draw_events(self):
        '''
        由于无法解决arrow的属性变换，故此实现采用每帧重画的方式
        主要实现在update函数：
            1、清空画布
            2、画底图
            3、根据id判断event
            4、画event
        '''
        fig_event, ax_event = plt.subplots()
        #pdb.set_trace()
        #ax_event.imshow(self.gd_img)

        #第一个注视event
        #base_sac = self.orderedEvents[0]
        #pdb.set_trace()
        #sac = plt.scatter(base_sac[3],base_sac[4],s=1)
        #pdb.set_trace()


        #base_sac = self.gaze_events[0]
        #sca, = plt.plot(base_sac[3], base_sac[4], 'ro',markersize=20)#初始点


        def update_events(i):#这个i是从0开始的，而event的frame不是
            #扫视如果迟迟没有得到更新就消失
            global event_idx
            #pdb.set_trace()
            cur_idx = i + self.start_frame
            '''纯注视
            if(event_idx>=len(self.gaze_events)):
                return sca
            if(cur_idx>=self.gaze_events[event_idx][1] and cur_idx <= self.gaze_events[event_idx][2]):#在当前event内
                sca.set_data(self.gaze_events[event_idx][3], self.gaze_events[event_idx][4])
            if(cur_idx==self.gaze_events[event_idx][2]):#到达event末尾
                event_idx += 1
            return sca
            '''
            #清空
            ax_event.cla()
            #画底图
            ax_event.imshow(self.gd_img)
            #判断event是否全部画完
            if(event_idx >= len(self.orderedEvents)):
                return
            #判断是否到达一个event
            if(cur_idx>=self.orderedEvents[event_idx][1] and cur_idx <= self.orderedEvents[event_idx][2]):
                if(self.orderedEvents[event_idx][0] == 0):#gaze
                    ga_event = self.orderedEvents[event_idx]
                    sca, = plt.plot(ga_event[3],ga_event[4],'ro',markersize=20)
                else:#saccade
                    sac_event = self.orderedEvents[event_idx]
                    x = sac_event[3]
                    y = sac_event[4]
                    dx = sac_event[5] - sac_event[3]
                    dy = sac_event[6] - sac_event[4]
                    ax_event.arrow(x,y,dx,dy,color='r',head_length=20,head_width=20)
            if(cur_idx==self.orderedEvents[event_idx][2]):#到达event末尾
                event_idx += 1


            '''#TODO 注视点逐渐变大，增加扫视箭头
            if(i>=self.orderedEvents[event_idx][1] and i <=self.orderedEvents[event_idx][2]):
                #if(self.orderedEvents[event_idx][0]==0):#gaze 画圆
                ga_event = self.orderedEvents[event_idx]
                #print(ga_event)
                area = 12**2#(i-ga_event[1])**2
                sac.set_offsets(ga_event[3],ga_event[4])#改坐标
                sac.set_sizes(area)#改面积
                if(i == ga_event[2]):#到达尾帧
                    event_idx += 1
                return sac
                #else:#saccade 画箭头
                    #return sac
            '''
        ani = FuncAnimation(fig=fig_event,
                            func=update_events,
                            frames=self.save_frames,
                            interval=33,
                            blit=False)

        ani.save(self.event_save_path, writer='ffmpeg', fps=30)
        print('Draw events finished！')

    def get_merge_ani(self, merge_save_path):
        self.draw_events()
        self.draw_gazePoints()
        final_clip = self.merge_ani()
        final_clip.write_videofile(merge_save_path)
        print('Merge Finished!')

    def merge_ani(self):
        #TODO: merge event ani and point ani to one video
        event_clip = VideoFileClip(self.event_save_path)
        point_clip = VideoFileClip(self.point_save_path)
        final_clip = clips_array([[event_clip],[point_clip]])

        return final_clip


    def get_events(self):#TODO: 得根据frame加event
        extractor = ga(self.ori_npData,conf.fixation_radius_threshold, conf.fixation_duration_threshold,
                       conf.saccade_min_velocity, conf.max_saccade_duration,self.eye_move_path)
        ori_gaze_events = extractor.get_gaze()
        ori_sac_events = extractor.get_sac()
        #get the event in the duration
        start_idx = 0
        end_idx = 0
        for i in range(len(ori_gaze_events)):
            if(ori_gaze_events[i][6]>=self.start_frame and start_idx==0):
                start_idx = i
            if(ori_gaze_events[i][7]>=self.end_frame):
                end_idx = i
                break
        ori_gaze_events = ori_gaze_events[start_idx:end_idx]#end_idx应该取不到，打算舍掉
        start_idx = 0
        end_idx = 0
        for i in range(len(ori_sac_events)):
            if(ori_sac_events[i][7]>=self.start_frame and start_idx==0):
                start_idx = i
            if(ori_sac_events[i][8]>=self.end_frame):
                end_idx = i
                break
        ori_sac_events = ori_sac_events[start_idx:end_idx]

        return ori_gaze_events, ori_sac_events

    def eventCoordTrans(self):
        '''
        coordinate trans and rebuild data structure
        0: event cls
        gaze:
            1: start frame
            2: end frame
            3: mean x
            4: mean y
        saccade:
            1: start frame
            2: end frame
            3: start x
            4: start y
            5: end x
            6: end y
            #7: peak vel
        :return:
        '''
        trans_ga_events = []
        trans_sca_events = []
        for ga in self.ori_gaze_events:#event并没有像gaze point那样坐标变化，故在此做坐标变换
            trans_ga = []
            trans_ga.append(0)# cls
            trans_ga.append(ga[6])
            trans_ga.append(ga[7])
            trans_ga.append(ga[0]*self.img_width)
            trans_ga.append(ga[1]*self.img_height)
            trans_ga_events.append(trans_ga)
        for sca in self.ori_sac_events:
            trans_sca = []
            trans_sca.append(1)
            trans_sca.append(sca[7])
            trans_sca.append(sca[8])
            trans_sca.append(sca[0]*self.img_width)
            trans_sca.append(sca[1] * self.img_height)
            trans_sca.append(sca[2] * self.img_width)
            trans_sca.append(sca[3] * self.img_height)
            trans_sca_events.append(trans_sca)
        return trans_ga_events, trans_sca_events


    def get_orderedEvents(self):
        '''
        融合gaze event list和 saccade event list
        :return:
        '''
        event_list = []
        num_gaze = len(self.gaze_events)
        num_sac = len(self.sac_events)
        ge = self.gaze_events
        se = self.sac_events
        gaze_idx = 0
        sac_idx = 0
        while(gaze_idx<num_gaze and sac_idx<num_sac):
            if(ge[gaze_idx][1]<se[sac_idx][1]):
                event_list.append(ge[gaze_idx])
                gaze_idx += 1
            else:
                event_list.append(se[sac_idx])
                sac_idx += 1
        if(gaze_idx==num_gaze):
            event_list.extend(se[sac_idx:])
        else:
            event_list.extend(ge[gaze_idx:])
        #pdb.set_trace()
        return event_list

    def get_npData(self):
        with open(self.eye_move_path, 'r') as f:
            json_data = json.load(f)
        eye_data = json_data["gazePoints"]
        ori_npData = self.json2np(eye_data)
        npData = ori_npData.copy()
        #transform gazePoints to the real coordinate
        npData[:,1] = npData[:,1]*self.img_width
        npData[:,2] = npData[:,2]*self.img_height
        return ori_npData, npData

    def json2np(self, eye_data):
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

    def get_gd(self):
        return plt.imread(self.gd_path)

event_idx = 0
#config
start_time = 187
end_time = 233
start_frame = start_time*30
end_frame = end_time*30
img_width = 1102
img_height = 620
save_frames = end_frame - start_frame

eye_move_path = 'data/JsonData/eye_576.json'
gd_img_path = 'data/gd_img/Stroop_1_1102x620.png'
event_save_name = 'output/Stroop1_event.mp4'
point_save_name = 'output/Stroop1_point.mp4'
merge_save_name = 'output/Stroop1_merged.mp4'
#gd_img_path = 'data/gd_img/Stroop_1_1102x620.png'
#gd image
if __name__ == '__main__':
    plot = ploter(eye_move_path,start_time,end_time,img_width,img_height,gd_img_path,event_save_name, point_save_name)
    #plot.draw_gazePoints()
    plot.get_merge_ani(merge_save_name)