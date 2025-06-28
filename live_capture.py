import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.tasks.python.core.base_options import BaseOptions
import cv2
from video_data import Video
from drawing_utils import drawing_utils
from detection import MediapipeDetect
import json5

mp_drawing = solutions.drawing_utils
mp_pose = mp.solutions.pose

def live_capture():

    try:
        with open('config.json5', 'r') as f:
            config = json5.load(f)
    
        video_path = config['videos']['input_path']
        model_path = config['models']['pose_detection']['light_model_path']
    except FileNotFoundError or PermissionError: #add more robust error handling 
        print('using defaults, config load failed')
        model_path = r"model\pose_landmarker_lite.task"
        video_path = r"vids\man_running.mp4"

    vid = cv2.VideoCapture(0)
#    vid.set(6,200)
#    vid.set(8,200)

    while(True):
        ret,frame = vid.read()
        print(ret)
        
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detection_result_landmarks = MediapipeDetect.return_landmarks(model_path= model_path,frame=frame)
        data = detection_result_landmarks
        annotated_frame = drawing_utils.draw_landmarks_on_image(rgb_frame, data)
        
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('current frame', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


live_capture()