import pandas as pd
import numpy as np
import json
import datetime


class Area(object):

    def __init__(self, duration_of_video, rec, last_frame_num):
        # H__GP100018.MP4_total_10800.csv
        self.rectangle = np.array(rec)
        # self.rectangle =  [200, 330, 1200, 550]

        self.ingress = 0
        self.egress = 0
        self.stay = 0
        self.stay_id = {}
        self.work_frame = None

        self.in_area_flag = 0
        self.out_area_flag = 0
        self.in_area_first_frame = float('inf')
        self.in_area_last_frame = 0
        self.out_area_first_frame = float('inf')
        self.out_area_last_frame = 0
        self.duration_of_video = duration_of_video

        self.last_frame_num = last_frame_num

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

        if self.count_stay():
            # print('stay')
            self.work_frame.apply(self.track_path, axis=1)
            if self.in_area_flag and self.out_area_flag:
                if self.in_area_first_frame > self.out_area_last_frame:
                    self.ingress += 1
                elif self.out_area_first_frame > self.in_area_last_frame:
                    pass
                else:
                    self.ingress += 1
            pass
        else:
            self.work_frame.apply(self.track_path, axis=1)
            if self.in_area_flag and self.out_area_flag:
                if self.in_area_first_frame > self.out_area_last_frame:
                    self.ingress += 1
                elif self.out_area_first_frame > self.in_area_last_frame:
                    self.egress += 1
                else:
                    self.ingress += 1
                    self.egress += 1

    def count_stay(self):
        tmp_frame = self.work_frame[(self.work_frame['frame_number'] == self.last_frame_num)|(self.work_frame['frame_number'] == self.last_frame_num-1)]
        if self.last_frame_num<self.work_frame.frame_number.max():

            print(self.last_frame_num)
            print(self.work_frame.frame_number.max())
            print('nononono')
        if tmp_frame.shape[0] > 0:
            object_id = tmp_frame['object_id'].unique()[0]
            self.stay_id[object_id] = 1
            self.stay = len(self.stay_id)
            return True
        else:
            return False

    def track_stay(self, row):
        x = (row.x1 + row.x2) / 2
        y = (row.y1 + row.y2) / 2
        if len(self.rectangle.shape) == 1:
            if self.in_area(x, y, self.rectangle):
                self.stay += 1
                return True
        else:
            flag = False
            for rec in self.rectangle:
                if self.in_area(x, y, rec):
                    flag = True
            if flag:
                self.stay += 1
                return True
        return False

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
    def in_area(self, x, y, rec):
        if x > rec[0] and x < rec[2] and y > rec[1] and y < rec[3]:
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


def count_set(work_frame, object, length_of_video, rec, last_frame_num):
    counter = Area(length_of_video, rec, last_frame_num)
    # print(work_frame)
    # print(object)
    object_frame = work_frame[work_frame['object_name'] == object]
    id_list = object_frame.object_id.unique()
    for i in id_list:
        tmp_frame = object_frame[object_frame['object_id'] == i]
        counter.set_work_frame(tmp_frame)
        counter.get_state()

    counter_frame = counter.get_results(object)
    return counter_frame


def main(work_frame, fps, length_of_video=4500, location='1', start_time=0, end_time=0, last_frame_num=9000):
    # 30fps. 9000
    # work_frame.to_csv(start_time+'.csv', index=False)
    # end_time_datetime = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    # start_time_datetime = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    # print(end_time)


    # print(location)
    # print(start_time)
    # print(end_time)
    with open('config.json') as f:
        location_dict = json.load(f)
    rec = location_dict[location]['person']
    motor_veh = location_dict[location]['veh']

    rec_matrix = {'person': rec, 'car': motor_veh, 'truck': motor_veh, 'bus': motor_veh,
                  'bicycle': motor_veh}
    total_frame = pd.DataFrame()

    # print('start counting')
    for obj in rec_matrix:
        object_frame = count_set(work_frame, obj, length_of_video, rec_matrix[obj], last_frame_num)
        total_frame = pd.concat([total_frame, object_frame], axis=1)

    # print('start saving')
    if start_time == 0:
        total_frame.to_csv(file_path[:-4] + '_count.csv', index=False)
    else:
        # total_frame['vehicle_ingress'] = total_frame['car_ingress'] + total_frame['bus_ingress'] + total_frame[
        #     'truck_ingress']
        # total_frame['vehicle_egress'] = total_frame['car_egress'] + total_frame['bus_egress'] + total_frame[
        #     'truck_egress']
        # total_frame['vehicle_stay'] = total_frame['car_stay'] + total_frame['bus_stay'] + total_frame['truck_stay']
        # total_frame = total_frame.filter(
        #     items=['person_ingress', 'person_egress', 'person_stay', 'vehicle_ingress', 'vehicle_egress',
        #            'vehicle_stay', 'bicycle_ingress', 'bicycle_egress', 'bicycle_stay'], axis=1)
        total_frame['location'] = location
        total_frame['start_time'] = start_time
        total_frame['end_time'] = end_time
        count_results_path =('count_results/location-' + location + '-' + start_time + '.csv').replace(' ','-').replace(':','-')
        print(count_results_path)
        total_frame.to_csv(count_results_path, index=False)
    # print('all good')


if __name__ == '__main__':
    work_frame = pd.read_csv("GP200514_location_1.MP4_total_10800.csv")
    length_of_video = 10800
    location = '1'
    start_time = str(datetime.datetime.now())
    end_time = str(datetime.datetime.now())
    main(work_frame, length_of_video, location, start_time, end_time)
