import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2



def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def main():
    IMAGE_FILE = r'images\image.png'  # Use raw string
    model_path = r'C:\Users\ananj\project\pose_buddy\model\pose_landmarker_lite.task'

    # if os.path.exists(IMAGE_FILE):
    #     print('Loaded file:', IMAGE_FILE)
    # else:
    #     print("File not found!")
    #     return

    # Load image for pose detection
    image = cv2.imread(IMAGE_FILE)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    
    # LANDMARKER is does the main stuff link figuring out shit
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # Use MediaPipe image format
    mp_image = mp.Image.create_from_file(IMAGE_FILE)

    # Detect pose
    detection_result = landmarker.detect(mp_image)

    # Draw pose on image
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    cv2.imshow("Pose Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Show segmentation mask
    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    cv2.imshow("Segmentation Mask", visualized_mask.astype(np.uint8))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
