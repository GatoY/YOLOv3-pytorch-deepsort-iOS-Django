import pandas as pd
import numpy as np
import datetime
import os


def work_line():
    work_frame = pd.DataFrame()
    work_path = 'count_results'
    files_list = os.listdir(work_path)
    for file in files_list:
        tmp_frame = pd.read_csv(work_path + '/' + file)
        work_frame = pd.concat([work_frame, tmp_frame], axis=0)
    work_frame = work_frame.filter(
        items=['location', 'start_time', 'end_time', 'person_ingress', 'person_egress', 'person_stay',
               'car_ingress', 'car_egress', 'car_stay', 'bus_ingress', 'bus_egress', 'bus_stay', 'truck_ingress',
               'truck_egress', 'truck_stay', 'bicycle_ingress', 'bicycle_egress', 'bicycle_stay'], axis=1)
    work_frame.sort_values(['location', 'start_time'], inplace=True)
    work_frame.to_csv('finals.csv', index=False)


if __name__ == '__main__':
    work_line()
    print('done')
