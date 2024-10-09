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
import json


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
        default="cpu", type=str)

    args = parser.parse_args()
    return args


class FaceStatusUpdater:
    def __init__(self):
        self.window_size = 5
        self.pitch_window = {}
        self.yaw_window = {}
        self.gaze_duration = {}
        self.previous_pitch = {}
        self.previous_yaw = {}
        self.previous_time = {}
        
    def update_face_status(self, face_id, pitch, yaw, location, frame_time, pitch_threshold=25, yaw_threshold=-25):
        if face_id not in self.pitch_window:
            self.pitch_window[face_id] = deque(maxlen=self.window_size)
            self.yaw_window[face_id] = deque(maxlen=self.window_size)
            self.gaze_duration[face_id] = 0
            self.previous_pitch[face_id] = float(np.degrees(pitch))
            self.previous_yaw[face_id] = float(np.degrees(yaw))
            self.previous_time[face_id] = current_time
        
        pitch_deg = float(np.degrees(pitch))
        yaw_deg = float(np.degrees(yaw))
        
        self.pitch_window[face_id].append(pitch_deg)
        self.yaw_window[face_id].append(yaw_deg)

        avg_pitch = float(np.mean(self.pitch_window[face_id]))
        avg_yaw = float(np.mean(self.yaw_window[face_id]))
        
        time_diff = current_time - self.previous_time[face_id]
        pitch_speed = abs(pitch_deg - self.previous_pitch[face_id]) / time_diff if time_diff > 0 else 0
        yaw_speed = abs(yaw_deg - self.previous_yaw[face_id]) / time_diff if time_diff > 0 else 0

        sight_speed = max(pitch_speed, yaw_speed)
        
        self.previous_pitch[face_id] = pitch_deg
        self.previous_yaw[face_id] = yaw_deg
        self.previous_time[face_id] = current_time
        
        gaze = abs(avg_pitch) < pitch_threshold and 15 > avg_yaw > yaw_threshold

        if gaze:
            self.gaze_duration[face_id] += frame_time    # Assume +1 frame = 1 
        else:
            self.gaze_duration[face_id] = 0
            
            
        face_status = {
            "face_id": face_id,
            "face_position": [int((location[0] + location[2]) / 2), int((location[1] + location[3]) / 2)],
            "face_size": [int(location[2] - location[0]), int(location[3] - location[1])],
            "sight_speed": float(round(sight_speed,2)),
            "gaze": gaze,
            "gaze_duration": round(self.gaze_duration[face_id],2)
        }

        return face_status


def gaze_tracking(video_path=None, interval=0.5, device='cpu', arch='ResNet50', display_output=True):
    cudnn.enabled = True

    gaze_pipeline = Pipeline(
        weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
        arch=arch,
        device=select_device(device, batch_size=1)
    )

    face_status_updater = FaceStatusUpdater()

    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)  # Use camera if video path is not provided

    if not cap.isOpened():
        raise IOError("Cannot open webcam or video")
    
    last_update_time = time.time()
    previous_time = time.time()

    with torch.no_grad():
        face_id_counter = 0
        while True:
            success, frame = cap.read()
            start_fps = time.time()

            if not success:
                print("Failed to obtain frame")
                time.sleep(0.1)
                continue

            current_time = time.time()
            frame_time = current_time - previous_time
            previous_time = current_time

            results = gaze_pipeline.step(frame)
            if results.pitch is not None:
                face_status_list = []
                for i in range(len(results.pitch)):
                    pitch = results.pitch[i]
                    yaw = results.yaw[i]
                    location = results.bboxes[i]

                    face_status = face_status_updater.update_face_status(
                        face_id_counter + i, pitch, yaw, location, frame_time)
                    face_status_list.append(face_status)

                if current_time - last_update_time >= interval:
                    for face_status in face_status_list:
                        print(json.dumps(face_status, indent=4))

                    last_update_time = current_time

            if display_output:
                frame = render(frame, results)
                myFPS = 1.0 / (time.time() - start_fps)
                cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
    
    

if __name__ == '__main__':  
    args = parse_args()
    cudnn.enabled = True

    gaze_pipeline = Pipeline(
        weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device = select_device(args.device, batch_size=1)
    )
    
    face_status_updater = FaceStatusUpdater()
        
    video_path = 'video/video02.mp4'
    cap = cv2.VideoCapture(video_path)
    #camera_url = 'http://192.168.10.245/leimCam/20240904/10'
    #cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    interval = 0.5   # define the interval
    last_update_time = time.time() 
    previous_time = time.time()
        
    
    with torch.no_grad():
        face_id_counter = 0
        while True:
            # Get frame
            success, frame = cap.read()    
            start_fps = time.time()  
        
            # Process frame
            current_time = time.time()
            frame_time = current_time - previous_time
            previous_time = current_time
               
            if not success:
                print("Failed to obtain frame")
                time.sleep(0.1)
            
            results = gaze_pipeline.step(frame)
            if results.pitch is not None:
                face_status_list = [] 
                for i in range(len(results.pitch)):
                    pitch = results.pitch[i]
                    yaw = results.yaw[i]
                    location = results.bboxes[i]

                    face_status = face_status_updater.update_face_status(face_id_counter + i, pitch, yaw, location, frame_time)
                    face_status_list.append(face_status) 
                    
                if current_time - last_update_time >= interval:
                    for face_status in face_status_list:
                        print(json.dumps(face_status, indent=4))  

                    last_update_time = current_time 
    
            frame = render(frame, results)
            
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                        
    cap.release()
    cv2.destroyAllWindows()
    