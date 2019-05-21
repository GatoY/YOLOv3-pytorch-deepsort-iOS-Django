import cv2
import pandas as pd
from utils_add import *
import numpy as np


def demo():
    work_frame = pd.read_csv(
        '/Users/Jason/Desktop/Fusion/yolo3_deepsort/enhance_post/results/GP200514_location_1.MP4_total_10800.csv')

    cap = cv2.VideoCapture('/Users/Jason/Desktop/Fusion/yolo3_deepsort/enhance_post/results/location_1_GP200514.MP4')
    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1)
    count = 0
    work_frame.frame_number = work_frame.frame_number.astype(int)

    while True:

        res, frame = cap.read()
        count = count + 1
        if res:

            rec = [[500, 282, 1138, 578], [1072, 161, 1168, 387]]
            motor_veh = [189, 121, 597, 496]

            if len(np.array(rec).shape) == 1:
                frame = cv2.rectangle(frame, (rec[0], rec[1]), (rec[2], rec[3]), (255, 0, 0), 1)
            else:
                for i in rec:
                    frame = cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), (255, 0, 0), 1)

            # motor_veh = [80, 200, 710, 600]
            if len(np.array(motor_veh).shape) == 1:
                frame = cv2.rectangle(frame, (motor_veh[0], motor_veh[1]), (motor_veh[2], motor_veh[3]), (0, 0, 255), 1)

            else:
                for i in motor_veh:
                    frame = cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 1)

            # tmp_frame = work_frame[work_frame.frame_number == count]
            tmp_frame = work_frame[work_frame.frame_number == (count + 1) / 2]
            positions = []
            tmp_frame.apply(
                lambda row: positions.append([row['x1'], row['y1'], row['x2'], row['y2'], row['object_id']]),
                axis=1)
            frame = plot_boxes_cv2(frame, positions)
            cv2.imshow('', frame)
            cv2.waitKey(1)

        else:
            break


cv2.destroyAllWindows()

############################################
if __name__ == '__main__':
    demo()
