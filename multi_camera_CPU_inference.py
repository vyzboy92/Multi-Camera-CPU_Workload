#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import cv2
import numpy as np
import sys
import time
import logging as log
from imutils.video import WebcamVideoStream, FileVideoStream, FPS
from openvino.inference_engine import IENetwork, IEPlugin
from utils import config


# Container function to initialise OpenVINO models
def init_model(xml, bins):
    model_xml = xml
    model_bin = bins
    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device='CPU')
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)

    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [
            l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_nets = plugin.load(network=net, num_requests=2)
    n, c, h, w = net.inputs[input_blob].shape
    del net
    return exec_nets, n, c, w, h, input_blob, out_blob, plugin


def frame_stack(frames):
    num_frames = len(frames)
    if num_frames == 1:
        return frames[0]
    if num_frames == 2:
        hstack1 = np.hstack((frames[0],frames[1]))
        return hstack1
    if num_frames == 3:
        hstack1 = np.hstack((frames[0],frames[1]))
        hstack2 = np.hstack((frames[2],config.no_input))
        vstack = np.vstack((hstack1,hstack2))
        return vstack
    if num_frames == 4:
        hstack1 = np.hstack((frames[0],frames[1]))
        hstack2 = np.hstack((frames[2],frames[3]))
        vstack = np.vstack((hstack1,hstack2))
        return vstack


def main():
    # Caffe model intialization
    age_net = cv2.dnn.readNetFromCaffe(
        "utils/age_gender_models/deploy_age.prototxt",
        "utils/age_gender_models/age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(
        "utils/age_gender_models/deploy_gender.prototxt",
        "utils/age_gender_models/gender_net.caffemodel")

    # Declaring classes and variables
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    gender_list = ['Male', 'Female']

    # Paths to models
    face_xml = "utils/openvino_models/face-detection-adas-0001.xml"
    face_bin = "utils/openvino_models/face-detection-adas-0001.bin"
    mask_xml = "utils/openvino_models/mask_model.xml"
    mask_bin = "utils/openvino_models/mask_model.bin"
    face_detection_net, n_f, c_f, w_f, h_f, input_blob_f, out_blob_f, plugin_f = init_model(
        face_xml, face_bin)
    mask_detection_net, n_m, c_m, w_m, h_m, input_blob_m, out_blob_m, plugin_m = init_model(
        mask_xml, mask_bin)

    streams = []
    for video in config.video_inputs:
        streams.append(WebcamVideoStream(src=video).start())
        time.sleep(0.5)

    # Initialize variables
    frame_count = 0
    cur_request_id_f = 0
    next_request_id_f = 1
    cur_request_id_m = 0
    next_request_id_m = 1
    cur_request_id_a = 0
    next_request_id_a = 1
    output_frame = None
    mask_detection_res = None

    # AVG FPS Initialization
    fps_avg = FPS().start()
    while True:
        frames=[]
        for stream in streams:
            frame = stream.read()
            if frame is None:
                frame = config.no_input
            else:
                frame = np.asarray(frame)
                frame = cv2.resize(frame,config.frame_size)
            frames.append(frame)

        processed_frames = []
        for frame in frames:
            if frame is None:
                break
            initial_h, initial_w = frame.shape[:2]
            # Find all the faces and face encodings in the current frame of video
            face_locations = []

            in_frame = cv2.resize(frame, (w_f, h_f))
            # Change data layout from HWC to CHW
            in_frame = in_frame.transpose((2, 0, 1))
            in_frame = in_frame.reshape((n_f, c_f, h_f, w_f))
            face_detection_net.start_async(request_id=cur_request_id_f, inputs={
                                        input_blob_f: in_frame})
            if face_detection_net.requests[cur_request_id_f].wait(-1) == 0:
                face_detection_res = face_detection_net.requests[cur_request_id_f].outputs[out_blob_f]
                for face_loc in face_detection_res[0][0]:
                    if face_loc[2] > 0.5:
                        xmin = abs(int(face_loc[3] * initial_w))
                        ymin = abs(int(face_loc[4] * initial_h))
                        xmax = abs(int(face_loc[5] * initial_w))
                        ymax = abs(int(face_loc[6] * initial_h))
                        face_locations.append((xmin, ymin, xmax, ymax))

            for (left, top, right, bottom) in face_locations:
                # Cropping the Faces and running it through mask detection model
                crop_img = frame[top:bottom, left:right]
                processed_crop_img = cv2.resize(crop_img, (w_m, h_m))
                processed_crop_img = processed_crop_img.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                processed_crop_img = processed_crop_img.reshape((n_m, c_m, h_m, w_m))

                # Openvino Mask detection
                mask_detection_net.start_async(request_id=cur_request_id_m, inputs={input_blob_m: processed_crop_img})
                if mask_detection_net.requests[cur_request_id_m].wait(-1) == 0:
                    mask_detection_res = mask_detection_net.requests[cur_request_id_m].outputs[out_blob_m]

                # Caffe Age and Gender Prediction
                blob2 = cv2.dnn.blobFromImage(
                    crop_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                # Predict Gender
                gender_net.setInput(blob2)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                # Predict age
                age_net.setInput(blob2)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]

                # Drawing bounding boxes
                frame = cv2.rectangle(frame, (left, top),
                                    (right, bottom), (255, 255, 255), 2)
                # Drawing Labels

                if mask_detection_res[0][0] > 0.5:
                    cv2.putText(frame, 'Masked: {:.2f}%'.format(mask_detection_res[0][0]*100), (left+15, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
                else:
                    cv2.putText(frame, 'Non-Masked: {:.2f}%'.format(mask_detection_res[0][1]*100), (left+15, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,12,255), 2)
                cv2.putText(frame, 'Age: {}'.format(age), (left+15, top+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
                cv2.putText(frame, 'Gender: {}'.format(
                    gender), (left+15, top+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

            fps_avg.stop()
            cv2.putText(frame, 'AVG_FPS: {:.2f}'.format(
                fps_avg.fps()), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 2)

            processed_frames.append(frame)
            frame_count += 1

            cur_request_id_f, next_request_id_f = next_request_id_f, cur_request_id_f
            cur_request_id_m, next_request_id_m = next_request_id_m, cur_request_id_m
            cur_request_id_a, next_request_id_a = next_request_id_a, cur_request_id_a
            
            output_frame = frame_stack(processed_frames)
        cv2.imshow('Output',output_frame)
        fps_avg.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('\nAVG_FPS: {}\n'.format(fps_avg.fps()))
    cv2.destroyAllWindows()
    for stream in streams:
        stream.stop()


if __name__ == '__main__':
    main()
