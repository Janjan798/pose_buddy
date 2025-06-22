import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.core.base_options import BaseOptions
import cv2

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in pose_landmarks])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def main():
    model_path = r"model\pose_landmarker_lite.task"
    video_path = r"vids\man_running.mp4"

    # Load MediaPipe pose detector
    base_options = BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    pose_detector = vision.PoseLandmarker.create_from_options(options)

    # Open input video
    cap = cv2.VideoCapture(video_path)

    # Set up output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output_with_pose.mp4", fourcc, 30.0, 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Create resizable display window
    cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pose Detection', 960, 540)  # Optional size

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run pose detection
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = pose_detector.detect(mp_image)

        # Draw landmarks
        annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)

        # Convert back to BGR for OpenCV display/write
        bgr_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Write frame to output
        out.write(bgr_annotated)

        # Display frame
        cv2.imshow('Pose Detection', bgr_annotated)

        # Exit on 'q' key or manual close
        key = cv2.waitKey(1)
        if key == ord('q') or cv2.getWindowProperty('Pose Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
