import pandas as pd
import numpy as np
import json
import datetime

ACCUMULATION_PERSON = 0
ACCUMULATION_CAR = 0
ACCUMULATION_BUS = 0
ACCUMULATION_TRUCK = 0
ACCUMULATION_BICYCLE = 0


def accumulate(row):
    acc_person = row['person_ingress'] - row['person_egress'] - row['person_stay']
    acc_car = row['car_ingress'] - row['car_egress'] - row['car_stay']
    acc_bus = row['bus_ingress'] - row['bus_egress'] - row['bus_stay']
    acc_truck = row['truck_ingress'] - row['truck_egress'] - row['truck_stay']
    acc_bicycle = row['bicycle_ingress'] - row['bicycle_egress'] - row['bicycle_stay']
    # global ACCUMULATION_PERSON
    # global ACCUMULATION_CAR
    # global ACCUMULATION_BUS
    # global ACCUMULATION_TRUCK
    # global ACCUMULATION_BICYCLE
    # ACCUMULATION_PERSON += acc_person
    # ACCUMULATION_CAR += acc_car
    # ACCUMULATION_BUS += acc_bus
    # ACCUMULATION_TRUCK += acc_truck
    # ACCUMULATION_BICYCLE += acc_bicycle

    # print(ACCUMULATION_BICYCLE)
    # print(ACCUMULATION_TRUCK)
    # print(ACCUMULATION_BUS)
    # print(ACCUMULATION_CAR)
    # print(ACCUMULATION_PERSON)
    num = 3
    if acc_person < -num:
        row['person_ingress'] += abs(acc_person)
        acc_person = 0
    elif acc_person > num:
        row['person_egress'] += acc_person
        acc_person = 0

    if acc_car < -num:
        row['car_ingress'] += abs(acc_car)
        acc_car = 0
    elif acc_car > num:
        row['car_egress'] += acc_car
        acc_car = 0

    if acc_bus < -num:
        row['bus_ingress'] += abs(acc_bus)
        acc_bus = 0
    elif acc_bus > num:
        row['bus_egress'] += acc_bus
        acc_bus = 0

    if acc_truck < -num:
        row['truck_ingress'] += abs(acc_truck)+1
        acc_truck = 0
    elif acc_truck > num:
        row['truck_egress'] += acc_truck -1
        acc_truck = 0

    if acc_bicycle < -num:
        row['bicycle_ingress'] += abs(acc_bicycle) + 1
        acc_bicycle = 0
    elif acc_bicycle > num:
        row['bicycle_egress'] += acc_bicycle -1
        acc_bicycle = 0

    return row


def main():
    work_frame = pd.read_csv('finals.csv')

    work_frame = work_frame.apply(accumulate, axis=1)

    work_frame.to_csv('test.csv')


if __name__ == '__main__':
    main()
