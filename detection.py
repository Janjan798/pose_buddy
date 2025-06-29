import mediapipe as mp
import numpy as np
from abc import ABC, abstractmethod
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.tasks.python.core.base_options import BaseOptions
import cv2


class Detect(ABC):

    @abstractmethod
    def return_landmarks():
        """returns a list of landmark points on a body with their
            coordinates in x,y """
        pass

class MediapipeDetect(Detect):
    def return_detection_result(model_path:str,frame:np.ndarray):
        '''returns the mediapipe vision.PoseLandmarker.detect() object'''
        # Load MediaPipe pose detector
        base_options = BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False)
        pose_detector = vision.PoseLandmarker.create_from_options(options)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = pose_detector.detect(mp_image)    
        
        return detection_result
    
    def return_landmarks(model_path:str,frame:np.ndarray) -> list:
        """returns a list of landmark points on a body with their
            coordinates in x,y,z for a single frame (the frame should be in BRG)"""
        detection_result = MediapipeDetect.return_detection_result(model_path,frame)    
        
        return detection_result.pose_landmarks,detection_result.pose_world_landmarks

    '''  
    def return_world_landmarks(model_path:str,frame:np.ndarray) -> list:
        """returns a list of landmark points in meters on a body with their
            coordinates in x,y,z for a single frame (the frame should be in BRG)"""
        detection_result = MediapipeDetect.return_detection_result(model_path,frame)    

        return detection_result.world_landmarks
'''
