import pandas as pd
import numpy as np


class Area(object):

    def __init__(self, duration_of_video):
        # H__GP100018.MP4_total_10800.csv
        self.rectangle = [440, 260, 1240, 680]
        # self.rectangle =  [200, 330, 1200, 550]

        self.ingress = 0
        self.egress = 0
        self.stay = 0
        self.work_frame = None

        self.in_area_flag = 0
        self.out_area_flag = 0
        self.in_area_first_frame = float('inf')
        self.in_area_last_frame = 0
        self.out_area_first_frame = float('inf')
        self.out_area_last_frame = 0
        self.duration_of_video = duration_of_video

    def set_work_frame(self, new_frame):
        self.work_frame = new_frame
        # print (self.work_frame)

    def get_state(self):

        self.in_area_flag = 0
        self.out_area_flag = 0

        self.in_area_first_frame = float('inf')
        self.in_area_last_frame = 0
        self.out_area_first_frame = float('inf')
        self.out_area_last_frame = 0

        if self.stay_or_not():
            # print('stay')
            pass
        else:
            self.work_frame.apply(self.track_path, axis=1)
            if self.in_area_flag and self.out_area_flag:
                # print('in fisrt' + str(self.in_area_first_frame))
                # print('in last' + str(self.in_area_last_frame))
                # print('out first' + str(self.out_area_first_frame))
                # print('out last' + str(self.out_area_last_frame))
                if self.in_area_first_frame > self.out_area_last_frame:
                    self.ingress += 1
                    # print('ingress')
                    # print('id' + str(self.work_frame.object_id.unique()))
                elif self.out_area_first_frame > self.in_area_last_frame:
                    self.egress += 1
                    # print ('egress')
                    # print('id' + str(self.work_frame.object_id.unique()))

                else:
                    self.ingress += 1
                    self.egress += 1

    def stay_or_not(self, threshold=30):
        if self.work_frame.frame_number.min() <= threshold and self.work_frame.frame_number.max() >= self.duration_of_video - threshold:
            self.stay += 1
            return 1
        else:
            return 0

    def track_path(self, row):
        x = (row.x1 + row.x2) / 2
        y = (row.y1 + row.y2) / 2

        if x > self.rectangle[0] and x < self.rectangle[2] and y > self.rectangle[1] and y < self.rectangle[3]:
            self.in_area_flag = 1
            if self.in_area_first_frame > row.frame_number:
                self.in_area_first_frame = row.frame_number
            if self.in_area_last_frame < row.frame_number:
                self.in_area_last_frame = row.frame_number
        else:
            self.out_area_flag = 1

            if self.out_area_first_frame > row.frame_number:
                self.out_area_first_frame = row.frame_number
            if self.out_area_last_frame < row.frame_number:
                self.out_area_last_frame = row.frame_number

    # return 1 if in rectangle or 0
    def in_area(self, x, y):
        if x > self.rectangle[0] and x < self.rectangle[2] and y > self.rectangle[1] and y < self.rectangle[3]:
            return 1
        else:
            return 0

    def print_results(self, name):
        print(name+'ingress number is: ' + str(self.ingress))
        print(name+'egress number is: ' + str(self.egress))
        print(name+'stay number is: ' + str(self.stay))


class Motor_Recorder(object):

    def __init__(self, duration_of_video):
        self.ingress = 0
        self.egress = 0
        self.stay = 0
        self.work_frame = None
        self.duration_of_video = duration_of_video
        self.first_frame = None
        self.last_frame = None

    def set_work_frame(self, new_frame):
        self.work_frame = new_frame
        self.first_frame = self.work_frame.frame_number.min()
        self.last_frame = self.work_frame.frame_number.max()

    def count(self, threshold):
        if self.stay_or_not():
            pass
        else:
            duration = self.last_frame - self.first_frame
            if duration > threshold:
                self.ingress += 1
                self.egress += 1

    def stay_or_not(self, threshold=30):
        if self.first_frame <= threshold and self.last_frame >= self.duration_of_video - threshold:
            self.stay += 1
            return 1
        else:
            return 0

    def print_results(self, name):
        print(name+' ingress number is: ' + str(self.ingress))
        print(name+' egress number is: ' + str(self.egress))
        print(name+' stay number is: ' + str(self.stay))


def motor_count_set(work_frame, object, length_of_video, threshold):
    counter = Motor_Recorder(length_of_video)
    object_frame = work_frame[work_frame['object_name'] == object]
    id = object_frame.object_id.unique()
    for i in id:
        tmp_frame = object_frame[object_frame['object_id'] == i]
        counter.set_work_frame(tmp_frame)
        counter.count(threshold)
    counter.print_results(object)


if __name__ == '__main__':
    # data_path = '/Users/Jason/Desktop/Fusion/yolo3_deepsort/results/H__GP100018.MP4_tota.....l_10800.csv'
    # data_path = '/Users/Jason/Desktop/Fusion/yolo3_deepsort/results/H__GP090018.MP4_total_21600.csv'
    # data_path = '/Users/Jason/Desktop/Fusion/yolo3_deepsort/results/F_result_GP080030.MP4_total_21600.csv'
    data_path = '/Users/Jason/Desktop/Fusion/yolo3_deepsort/results/results.csv'

    length_of_video = 21600
    work_frame = pd.read_csv(data_path)

    area_counter = Area(length_of_video)
    person_frame = work_frame[work_frame['object_name'] == 'person']
    trackers_id = work_frame.object_id.unique()
    for i in trackers_id:
        # get frame of object i.
        tmp_frame = person_frame[person_frame['object_id'] == i]
        area_counter.set_work_frame(tmp_frame)
        area_counter.get_state()
        # print(area_counter.print_results())

    objects_threshold_list = {'car': 90, 'truck': 90, 'bus': 90, 'bicycle': 90}
    for i in objects_threshold_list:
        motor_count_set(work_frame, i, length_of_video, objects_threshold_list[i])

    area_counter.print_results('people ')
