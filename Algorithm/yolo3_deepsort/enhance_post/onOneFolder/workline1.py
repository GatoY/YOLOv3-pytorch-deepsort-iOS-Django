import pandas as pd
import numpy as np
import datetime
from postprocess import main
import os
import multiprocessing


# FPS = 4500


# 1_2018-07-24-20-54-17_2018-07-24-20-56-49_4560.csv
def process(work_path, file, files_dict):
    work_frame = pd.read_csv(work_path + '/' + file)

    start_datetime = files_dict[file]['start_datetime']
    end_datetime = files_dict[file]['end_datetime']
    location = files_dict[file]['location']
    length_of_video = files_dict[file]['length_of_video']
    fps = files_dict[file]['fps']

    if start_datetime.minute % 5 == 0 and start_datetime.second == 0:
        count = 0
        while (1):
            to_datetime = start_datetime + datetime.timedelta(minutes=(count + 1) * 5)
            if to_datetime < end_datetime:
                tmp_frame = work_frame[work_frame['frame_number'] > count * fps * 5 * 60]
                tmp_frame = tmp_frame[tmp_frame['frame_number'] <= (count + 1) * fps * 5 * 60]
                print(tmp_frame.shape[0])
                main(tmp_frame, fps, fps * 5 * 60, location,
                     str(start_datetime + datetime.timedelta(minutes=count * 5)),
                     str(to_datetime), (count + 1) * fps * 5 * 60)
                print(4)

                count += 1
            else:
                break
    else:
        diff = start_datetime.minute - int(start_datetime.minute / 5) * 5
        from_datetime = start_datetime - datetime.timedelta(minutes=diff) - datetime.timedelta(
            seconds=start_datetime.second)
        seconds_intotal = int((end_datetime - start_datetime).total_seconds())
        second_frame_length = int((from_datetime + datetime.timedelta(
            minutes=5) - start_datetime).total_seconds()) / seconds_intotal * length_of_video
        second_frame = work_frame[work_frame['frame_number'] <= second_frame_length]
        second_frame['frame_number'] = second_frame['frame_number'] + (fps * 5 * 60 - second_frame_length)
        makeup_flag = 0

        for i in files_dict:
            front_strart_datetime = files_dict[i]['start_datetime']
            front_end_datetime = files_dict[i]['end_datetime']

            if from_datetime >= front_strart_datetime and from_datetime <= front_end_datetime:
                first_frame_length = int(fps * 5 * 60 - second_frame_length)
                total_frame = pd.read_csv(work_path + '/' + i)
                first_frame = total_frame[
                    total_frame['frame_number'] >= int(files_dict[i]['length_of_video'] - first_frame_length)]
                diff_frame = first_frame['frame_number'].min() - 1
                first_frame['frame_number'] = first_frame['frame_number'] - diff_frame
                makeup_flag = 1
                break
        if makeup_flag == 1:
            tmp_frame = pd.concat([first_frame, second_frame], axis=0)
            main(tmp_frame, fps, fps * 5 * 60, location, str(from_datetime),
                 str(from_datetime + datetime.timedelta(minutes=5)), 9000)
            # print(3)
            #
            # print(tmp_frame.shape[0])


        else:
            tmp_frame = second_frame
            main(tmp_frame, fps, fps * 5 * 60, location, str(start_datetime),
                 str(from_datetime + datetime.timedelta(minutes=5)), 9000)
            # print(1)
            #
            # print(tmp_frame.shape[0])

        count = 0
        while (1):
            to_datetime = from_datetime + datetime.timedelta(minutes=(count + 2) * 5)
            if to_datetime < end_datetime:
                tmp_frame = work_frame[(work_frame['frame_number'] > (second_frame_length + count * fps * 5 * 60))|(work_frame['frame_number']<=(second_frame_length + (count + 1) * fps * 5 * 60))]
                # print((second_frame_length + (count + 1) * fps * 5 * 60)-(second_frame_length + count * fps * 5 * 60))
                # tmp_frame = tmp_frame[tmp_frame['frame_number'] <=
                main(tmp_frame, fps, fps * 5 * 60, location,
                     str(from_datetime + datetime.timedelta(minutes=(count + 1) * 5)),
                     str(to_datetime))
                # print('fps'+str(fps))
                # print((second_frame_length + (count + 1) * fps * 5 * 60) - (second_frame_length + count * fps * 5 * 60))


                count += 1
            else:
                break


def work_line():
    work_path = 'location1'
    files_list = os.listdir(work_path)
    files_dict = {}
    # first_time = datetime.datetime(2100, 01, 01)
    # last_time = datetime.datetime(1995, 03, 29)
    for file in files_list:
        files_list = file.split('_')
        location = str(int(files_list[0]))
        if location == '1':
            start_year, start_month, start_day, start_hour, start_min, start_sec = files_list[1].split('-')
            start_datetime = datetime.datetime(int(start_year), int(start_month), int(start_day), int(start_hour),
                                               int(start_min), int(start_sec))
            end_year, end_month, end_day, end_hour, end_min, end_sec = files_list[2].split('-')
            end_datetime = datetime.datetime(int(end_year), int(end_month), int(end_day), int(end_hour), int(end_min),
                                             int(end_sec))
            length_of_video = int(files_list[3].split('.')[0])

            fps = length_of_video / int((end_datetime - start_datetime).total_seconds())

            files_dict[file] = {'location': location, 'start_datetime': start_datetime, 'end_datetime': end_datetime,
                                'length_of_video': length_of_video, 'fps': fps}

    for file in files_dict:
        p = multiprocessing.Process(target=process, args=(work_path, file, files_dict,))
        p.start()
        # break


if __name__ == '__main__':
    work_line()
    print('done')
