import mediapipe as mp
from mediapipe import solutions
import cv2
from video_data import Video
from drawing_utils import drawing_utils
from detection import MediapipeDetect
import json5
from graphing_utils import GraphingUtils

mp_drawing = solutions.drawing_utils
mp.pose = solutions.pose
video1 = Video("video1")
grapher=GraphingUtils()


def main():
    try:
        with open('config.json5', 'r') as f:
            config = json5.load(f)
    
        video_path = config['videos']['input_path']
        model_path = config['models']['pose_detection']['full_model_path']
    except FileNotFoundError or PermissionError: #add more robust error handling 
        print('using defaults, config load failed')
        model_path = r"model\pose_landmarker_lite.task"
        video_path = r"vids\man_running.mp4"



    cap = cv2.VideoCapture(video_path) # makes video capture object i.e the Video

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("vids/output_with_pose.mp4", fourcc, 30.0, 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    

    frno=0 #frame no.
    # Create resizable display window
    cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pose Detection', 960, 540)  # Optional size

    while cap.isOpened(): # is the video frame loop 
        ret, frame = cap.read() # return the current feed from the Video  
        if not ret: # ret is a true false variable frame is the image from the 
            break
        
        frno+=1
        
        # POSE DETECTION BEGIN
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

     

        # Draw landmarks
        data,data3d = MediapipeDetect.return_landmarks(model_path= model_path,frame=frame)
        #writing data into list
        video1.data_into_list(data,frno)#landmarks are uploaded into the object's all_landmarks list
        video1.data3d_into_list(data3d,frno) 

        annotated_frame = drawing_utils.draw_landmarks_on_image(rgb_frame, data)

        # Convert back to BGR for OpenCV display/write
        bgr_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        #POSE DETECTION END

        # Write frame to output
        out.write(bgr_annotated)

        # Display frame
        cv2.imshow('Pose Detection', bgr_annotated)
       
        # Exit on 'q' key or manual close
        key = cv2.waitKey(1)# IDK WTF THIS DOES BUT 1 WORKS oh it waits for 1
        if key == ord('q') or cv2.getWindowProperty('Pose Detection', cv2.WND_PROP_VISIBLE) < 1:
            break


    video1.list_into_dframe()
    grapher.plot2d(video1,20)
    grapher.plot_landmark_with_time(video1,"x",26)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
