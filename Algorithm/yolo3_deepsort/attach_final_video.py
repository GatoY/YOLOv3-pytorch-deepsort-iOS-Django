import cv2
import pandas as pd
from utils_add import *


def demo():
    work_frame = pd.read_csv('results/results.csv')
    cap = cv2.VideoCapture('results/GP100018.MP4')
    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1)
    count = 0
    work_frame.frame_number = work_frame.frame_number.astype(int)
    res, frame = cap.read()
    print(frame.shape)
    while True:

        res, frame = cap.read()
        count = count + 1
        if count % 2 == 0:
            continue
        if res:
            print('frame: ' + str(count))
            tmp_frame = work_frame[work_frame.frame_number == (count+1)/2]
            positions = []
            tmp_frame.apply(lambda row: positions.append([row['x1'], row['y1'], row['x2'], row['y2'], row['unique_id']]), axis=1)
            draw_img = plot_boxes_cv2(frame, positions)
            cv2.imshow('', draw_img)
            cv2.waitKey(0)
        else:
            break
    cv2.destroyAllWindows()

############################################
if __name__ == '__main__':
    demo()
