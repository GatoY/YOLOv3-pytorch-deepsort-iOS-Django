import pandas as pd
import numpy as np
import multiprocessing
import time
import os


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


# location 1 person [500, 240, 1200, 600] veh [80, 200, 710, 600]  $
# location 2 person [440, 260, 1240, 680] veh [90, 260, 538, 613]  $
# location 3 person [[456, 284, 1105, 579], [305, 207, 380, 346]] veh [110, 146, 561, 487]  $
# location 4 person [[546, 333, 1120, 571], [408,237,546,333]] veh [155, 246, 624, 505]  $
# location 5 person [[530, 372, 1147, 548], [98, 276, 530, 376]] veh [200, 220, 500, 559] $
# location 6 person [[546, 333, 1200, 571], [85, 390, 436, 642]] veh [268, 338, 647, 528]  $
# location 7 person [[137, 327, 574, 569], [362, 260, 400, 327]] veh [689, 326, 1212, 436]  $
# location 8 person [624, 368, 1131, 487] veh [[240, 310, 395, 521], [134, 282, 160, 310], [134, 310, 240, 350]]  $
# location 9 person  [[20, 384, 500, 562], [1015, 289, 1236, 625], [776, 216, 965, 262], [271, 252, 516, 280]]
#            veh [[268, 244, 914, 540]]  $
# location 10 person [[268, 308, 594, 500], [131, 389, 268, 543]] veh [[976, 237, 1077, 314], [773, 314, 1125, 445]]  $
# location 11 person [[144, 370, 804, 552], [980, 350, 1120, 507], [707, 270, 962, 338], [350, 285, 450, 336]]
#             veh [[397, 335, 986, 560], [192, 328, 397, 638]]  $
# location 12 person [[136, 282, 697, 500], [432, 204, 513, 289]] veh [[559, 265, 959, 306], [427, 159, 494, 300]]  $

def main(file_path, location=2, start_time=0, end_time=0):
    # 30fps. 9000
    work_frame = pd.read_csv(file_path)
    length_of_video = 10800
    # location 7 person  veh   $

    rec = [[500, 282, 1138, 578], [1072, 161, 1168, 387]]
    motor_veh = [189, 121, 597, 496]
    rec_matrix = {'person': rec, 'car': motor_veh, 'truck': motor_veh, 'bus': motor_veh,
                  'bicycle': motor_veh}
    # rec_matrix = {'person': [440, 260, 1150, 600], 'car': motor_veh, 'truck': motor_veh, 'bus': motor_veh,'bicycle': motor_veh}
    total_frame = pd.DataFrame()
    for obj in rec_matrix:
        object_frame = count_set(work_frame, obj, length_of_video, rec_matrix[obj])
        total_frame = pd.concat([total_frame, object_frame], axis=1)
    if start_time == 0:
        total_frame.to_csv(file_path[:-4] + '_count.csv', index=False)
    else:
        total_frame['location'] = location
        total_frame['start_time'] = start_time
        total_frame['end_time'] = end_time
        total_frame.to_csv('location-' + location + start_time + '.csv', index=False)


if __name__ == '__main__':
    t = time.time()
    data_path = '/Users/Jason/Desktop/Fusion/yolo3_deepsort/enhance_post/results/'
    data_path_list = data_path + 'GP200514_location_1.MP4_total_10800.csv'
    main(data_path_list)
    # break
