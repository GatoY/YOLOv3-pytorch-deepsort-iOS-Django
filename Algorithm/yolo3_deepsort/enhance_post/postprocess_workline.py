import pandas as pd
import numpy as np
import datetime
import postprocess
from apscheduler.schedulers.blocking import BlockingScheduler
import traceback
import os

def work_line():
    work_path = '/'
    files_list = os.listdir(work_path)
    files_dict = {}
    first_time ='01-01-2100'
    last_time = '29-29-1995'
    for file in files_list:
        files_dict[file] = [file[:], file[:]]
        first_time = file[:] if file[:]<first_time else first_time
        last_time = file[:] if file[:] > last_time else last_time




    start_time = ''
    end_time = ''

    work_frame = ''
    postprocess.main(work_frame, location, start_time, end_time)


if __name__ == '__main__':
    scheduler = BlockingScheduler()
    # 8 am every day.
    # scheduler.add_job(main.detecting, 'cron', hour='8')
    scheduler.add_job(work_line(), 'interval', minutes=5)

    try:
        scheduler.start()
    except:
        # toUser = 'yu.liu@kepleranalytics.com.au'
        # msg = traceback.format_exc()
        # print(msg)
        # sendEmail(toUser, msg)
        pass
