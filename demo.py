import argparse
import pathlib
import numpy as np
import time
from collections import deque

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from PIL import Image, ImageOps

from face_detection import RetinaFace

from l2cs import select_device, draw_gaze, getArch, Pipeline, render
import cv2

CWD = pathlib.Path.cwd()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--device',dest='device', help='Device to run model: cpu or gpu:0',
        default="0", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet18', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    cam = args.cam_id
    # snapshot_path = args.snapshot

    gaze_pipeline = Pipeline(
        weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device = select_device(args.device, batch_size=1)
    )
    
    video_path = 'video/video02.mp4'
    cap = cv2.VideoCapture(video_path)
    #camera_url = 'http://192.168.10.245/leimCam/20240904/10'
    #cap = cv2.VideoCapture(camera_url)


    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    window_size = 5
    pitch_window = deque(maxlen=window_size)
    yaw_window = deque(maxlen=window_size)

    with torch.no_grad():
        while True:
            # Get frame
            success, frame = cap.read()    
            start_fps = time.time()  

            if not success:
                print("Failed to obtain frame")
                time.sleep(0.1)
            
            pitch_threshold = 25  
            yaw_threshold = -25
            

            # Process frame
            results = gaze_pipeline.step(frame)
            if results.pitch is not None:
                for i in range(len(results.pitch)):
                    pitch = results.pitch[i]
                    yaw = results.yaw[i]
                    location = results.bboxes[i]
                    center_x = int((location[0] + location[2]) / 2)
                    center_y = int((location[1] + location[3]) / 2)
                    center = [center_x, center_y]

                    pitch_deg = np.degrees(pitch)
                    yaw_deg = np.degrees(yaw)
                    
                    pitch_window.append(pitch_deg)
                    yaw_window.append(yaw_deg)

                    avg_pitch = np.mean(pitch_window)
                    avg_yaw = np.mean(yaw_window)

                    print(f"目标{i+1}: pitch: {avg_pitch:.3f}, yaw: {avg_yaw:.3f}, location: {center}")
                    
                    if abs(avg_pitch) < pitch_threshold and 15 > avg_yaw > yaw_threshold:
                        print(f"目标{i+1} 正在注视摄像头")
                    else:
                        print(f"目标{i+1} 未注视摄像头")
    
            frame = render(frame, results)
            
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            success,frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    