from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import argparse
import pickle as pkl

from sort import *

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


# def write(x, img):
def get_filtered_boxes(output_boxes,interested_objects):
    return_boxes = []
    labels = []
    for box in output_boxes:

        cls = int(box[-1])
        label = classes[cls]
        if label not in interested_objects:
            continue
        x1 = int(box[1])
        y1 = int(box[2])
        x2 = int(box[3])
        y2 = int(box[4])

        # c1 = (x1, y1)
        # c2 = (x2, y2)

        return_boxes.append([x1, y1, x2, y2])
        labels.append(label)
    return_boxes = np.array(return_boxes)
    # print(return_boxes)
    return return_boxes, labels
    # print(label)
    # color = random.choice(colors)
    # print(x1)

    # boxes.append([x1, y1, x2, y2, label])
    # cv2.rectangle(img, (x1,y1), (x2, y2), [0,255,0], 1)
    # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    # c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4

    # cv2.rectangle(img, c1, c2, color)
    # cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    # return img
    # print ('w_boxes' + str(w_boxes))
    # return w_boxes


def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.3)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="320", type=str)
    return parser.parse_args()


# def tracker_track(frame, boxs)

if __name__ == '__main__':
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80
    csv_file_path = 'test.csv'
    interested_objects = ['person', 'car', 'bus', 'truck']

    args = arg_parse()

    # This the threshold for confidence
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)

    CUDA = torch.cuda.is_available()

    classes = load_classes('data/coco.names')

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    cap = cv2.VideoCapture('results/GP090018.MP4')

    assert cap.isOpened(), 'Cannot capture source'

    count = 0
    start = time.time()

    mot_tracker = Sort()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img, orig_im, dim = prep_image(frame, inp_dim)

        if CUDA:
            img = img.cuda()

        output = model(Variable(img), CUDA)
        # print(confidence)

        output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        # Didn't detect anything
        if type(output) == int:
            boxes = np.array([])
            labels = np.array([])
        else:
            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim

            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]

            boxes, labels = get_filtered_boxes(output, interested_objects)

        mot_tracker.update(boxes, labels)
    try:
        mot_tracker.generate_csv(csv_file_path)

    except:
        print('FileName error')

