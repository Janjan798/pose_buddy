
import numpy as np
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions



class drawing_utils():

    def draw_landmarks_on_image(rgb_image, detection_result):
        pose_landmarks_list = detection_result
        annotated_image = np.copy(rgb_image)

        for pose_landmarks in pose_landmarks_list: #pose_landmarks_list is a list of people detected not landmarks 
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in pose_landmarks]) # adds normalised landmarks to each the pose_landmarks_proto list of which is an empty list of Normalised Landmarks
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()) # I assume this is the drawing tool
        return annotated_image

