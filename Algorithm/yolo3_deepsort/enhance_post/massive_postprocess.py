import pandas as pd
import numpy as np
import multiprocessing
import time
import os
import json


class Area(object):

    def __init__(self, duration_of_video, rec):
        # H__GP100018.MP4_total_10800.csv
        self.rectangle = np.array(rec)
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

    def stay_or_not(self, threshold=60):
        if self.work_frame.frame_number.min() <= threshold and self.work_frame.frame_number.max() >= self.duration_of_video - threshold:
            self.stay += 1
            return 1
        else:
            return 0

    def track_path(self, row):
        x = (row.x1 + row.x2) / 2
        y = (row.y1 + row.y2) / 2
        if len(self.rectangle.shape) == 1:
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
        else:
            for rec in self.rectangle:
                if x > rec[0] and x < rec[2] and y > rec[1] and y < rec[3]:
                    self.in_area_flag = 1
                    if self.in_area_first_frame > row.frame_number:
                        self.in_area_first_frame = row.frame_number
                    if self.in_area_last_frame < row.frame_number:
                        self.in_area_last_frame = row.frame_number
                    return

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
        print(name + ' ingress number is: ' + str(self.ingress))
        print(name + ' egress number is: ' + str(self.egress))
        print(name + ' stay number is: ' + str(self.stay))

    def get_results(self, name):
        new_frame = pd.DataFrame()
        new_frame[name + '_ingress'] = [self.ingress]
        new_frame[name + '_egress'] = [self.egress]
        new_frame[name + '_stay'] = [self.stay]
        return new_frame


def count_set(work_frame, object, length_of_video, rec):
    counter = Area(length_of_video, rec)
    # print('print' + str(object))
    # print(work_frame['object_name'])
    object_frame = work_frame[work_frame['object_name'] == object]
    id_list = object_frame.object_id.unique()
    for i in id_list:
        tmp_frame = object_frame[object_frame['object_id'] == i]
        counter.set_work_frame(tmp_frame)
        counter.get_state()
    # counter.print_results(object)
    counter_frame = counter.get_results(object)
    return counter_frame


# location 1 person [440, 260, 1240, 680] veh [90, 260, 538, 613]
# location 2 person [440, 260, 1240, 680] veh [90, 260, 538, 613]
# location 5 person [440, 260, 1150, 600] veh [60, 250, 538, 613]
def main(file_path, i, location_dict):
    # 30fps. 9000
    print(i)
    location = str(int(i.split('_')[2].split('.')[0]))
    print(location)
    rec = location_dict[location]['person']
    motor_veh = location_dict[location]['veh']
    work_frame = pd.read_csv(file_path + i)
    length_of_video = 4500

    rec_matrix = {'person': rec, 'car': motor_veh, 'truck': motor_veh, 'bus': motor_veh,
                  'bicycle': motor_veh}
    total_frame = pd.DataFrame()
    for obj in rec_matrix:
        object_frame = count_set(work_frame, obj, length_of_video, rec_matrix[obj])
        total_frame = pd.concat([total_frame, object_frame], axis=1)
    total_frame.to_csv('count_results/'+i + '_count.csv', index=False)


if __name__ == '__main__':
    t = time.time()
    data_path = '/Users/Jason/Desktop/Fusion/yolo3_deepsort/enhance_post/results/'

    data_path_list = os.listdir(data_path)

    with open('config.json') as f:
        location_dict = json.load(f)

    for i in data_path_list:
        p = multiprocessing.Process(target=main, args=(data_path, i, location_dict,))
        p.start()