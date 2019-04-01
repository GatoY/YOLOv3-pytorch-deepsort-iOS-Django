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

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_


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
def write(x):
    x1 = int(x[1])
    y1 = int(x[2])
    x2 = int(x[3])
    y2 = int(x[4])
    # c1 = (x1, y1)
    # c2 = (x2, y2)
    cls = int(x[-1])

    label = classes[cls]
    # print(label)
    # color = random.choice(colors)
    # print(x1)
    interested_objects = ['person', 'car', 'bus', 'truck']
    boxes = [x1, y1, x2, y2]
    # cv2.rectangle(img, (x1,y1), (x2, y2), [0,255,0], 1)
    # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    # c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4

    # cv2.rectangle(img, c1, c2, color)
    # cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    # return img
    return boxes


def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.35)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="160", type=str)
    return parser.parse_args()

# def tracker_track(frame, boxs)

if __name__ == '__main__':
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80

    #########################################################
    # deep_sort
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    #########################################################

    args = arg_parse()

    # This the threshold for confidence
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    num_classes = 80
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

    cap = cv2.VideoCapture('vtest.avi')

    assert cap.isOpened(), 'Cannot capture source'

    count = 0
    start = time.time()
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:

            img, orig_im, dim = prep_image(frame, inp_dim)

            if CUDA:
                img = img.cuda()

            output = model(Variable(img), CUDA)
            # print(confidence)

            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)
            # print(confidence)
            # print(output)
            # Didn't detect anything
            if type(output) == int:
                count += 1
                # print("FPS of the video is {:5.2f}".format(count / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim

            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]

            colors = pkl.load(open("pallete", "rb"))

            boxs = list(map(lambda x: write(x), output))
            # list(map(lambda x: write(x, frame), output))
            # print(boxs)

            #########################################################
            #
            features = encoder(orig_im, boxs)
            print(features)
            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if track.is_confirmed() and track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                cv2.rectangle(orig_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(orig_im, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

            for det in detections:
                bbox = det.to_tlbr()
                cv2.rectangle(orig_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

            #########################################################

            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(0)
            if key & 0xFF == ord('q'):
                break
            count += 1
            # print("FPS of the video is {:5.2f}".format(count / (time.time() - start)))
        else:
            break
