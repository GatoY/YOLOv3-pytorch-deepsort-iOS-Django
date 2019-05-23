from __future__ import division
import time
import torch
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import random
import pickle as pkl
import argparse
from recorder import Recorder

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

import warnings

warnings.filterwarnings("ignore")


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
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
    
    Returns a Variable git a
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img):
    c1 = tuple(x[1:3].int())  # left angle
    c2 = tuple(x[3:5].int())  # right angle
    cls = int(x[-1])  # classification
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img


def arg_parse():
    """
    Parse arguements to the detect module
    
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

    parser.add_argument("--video", dest='video', help="Video to run detection upon", default="video.avi", type=str)
    parser.add_argument("--dataset", dest="dataset", help="Dataset on which the network has been trained",
                        default="pascal")
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.6)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default="416",
                        type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()

    confidence = float(args.confidence)
    # TODO
    nms_thesh = float(args.nms_thresh)
    start = 0
    num_classes = 17
    CUDA = torch.cuda.is_available()

    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()
    model(get_test_input(inp_dim, CUDA), CUDA)
    model.eval()

    # videofile = args.video
    videofile = 'vtest.avi'
    cap = cv2.VideoCapture(videofile)
    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()
    print('while')

    classes = load_classes('data/coco.names')
    colors = pkl.load(open("pallete", "rb"))

    #############
    model_filename = 'data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    ###############
    records = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img, orig_im, dim = prep_image(frame, inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)
        if type(output) == int:
            frames += 1
            # print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
            # cv2.imshow("frame", orig_im)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            continue

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2
        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        detection_boxes = output.numpy()[:, (1, 2, 3, 4, -1)]
        # filter 17
        boxes = []
        labels = []
        for box in detection_boxes:
            if box[-1] <= 18:
                boxes.append(box[0:4])
                labels.append(classes[int(box[-1])])
        # np_output, [1:3] left angle, [3:5] right angle. [-1] classification

        # list(map(lambda x: write(x, orig_im), output))

        # TODO
        features = encoder(orig_im, boxes)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature, label) for bbox, feature, label in zip(boxes, features, labels)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # counts = {i: j for (i, j) in zip(classes, [0] * len(classes))}

        for track in tracker.tracks:
            if track.track_id not in records:
                records[track.track_id] = Recorder(track.track_id, track.label, frames)
            else:
                records[track.track_id].update(frames)
            # counts[track.label] += 1
            # if track.is_confirmed() and track.time_since_update > 1:
            #     continue
            # bbox = track.to_tlbr()
            # cv2.rectangle(orig_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            # cv2.putText(orig_im, str(track.label), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        # for det, lab in zip(detections, labels):
        #     bbox = det.to_tlbr()
        #     cv2.rectangle(orig_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        # print(counts)
        ###############
        # cv2.imshow("frame", orig_im)
        # key = cv2.waitKey(1)
        # if key & 0xFF == ord('q'):
        #     break
        frames += 1
        print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
    counts = {i: j for (i, j) in zip(classes, [0] * len(classes))}

    for id in records:
        recorder = records[id]
        result = recorder.count()
        print(result)
        if result != False:
            counts[result] += 1
    print('result is %s' % counts)
