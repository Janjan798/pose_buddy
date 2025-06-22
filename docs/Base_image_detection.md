* Arnav Yadavilli 2025

Specify image path
specify model path


set base_options as a Base_Options object from the mediapipe Python Class

set options as a vision.PoseLandmarkerOptions object from the mediapipe python vision class

image needs to be converted to a mediapipe image fomat with mp.Image.create_from_file-

Prepare data
Prepare your input as an image file or a numpy array, then convert it to a mediapipe.Image object. If your input is a video file or live stream from a webcam, you can use an external library such as OpenCV to load your input frames as numpy arrays.


pass the image as a argument to the pose detection function with detection_result = detector.detect(mp_image)

Our next step is to Draw the Detected pose onto the Image

we made our own function for that i.e draw_landmarks_on_image that takes two args rgb_image, detection_result


-
Fo Video
import mediapipe as mp

# Use OpenCV’s VideoCapture to load the input video.

# Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
# You’ll need it to calculate the timestamp for each frame.

# Loop through each frame in the video using VideoCapture#read()

# Convert the frame received from OpenCV to a MediaPipe’s Image object.
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
    

The Pose Landmarker uses the detect, detect_for_video and detect_async functions to trigger inferences. For pose landmarking, this involves preprocessing input data and detecting poses in the image.

# Perform pose landmarking on the provided single image.
# The pose landmarker must be created with the video mode.
pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    



