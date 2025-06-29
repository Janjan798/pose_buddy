import csv

class Video():
    '''Class for each video, containing its own all_landmarks list and a video title as attributes 
    '''
    def __init__(self,name="video"):
        self.all_landmarks = []
        self.all_3dlandmarks=[]
        self.name=name

    def data_into_list(self,data:list,frno):
        '''Takes data from mediapipe object and appends it to all_landmarks list of that object
        '''
        for pose in data:
            count=0
            for lm in pose:
                self.all_landmarks.append({"frame":frno,
                    "landmark":count,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility,
                    "presence": lm.presence
                })
                count+=1

    def data3d_into_list(self,data:list,frno):
        '''Takes data from mediapipe object and appends it to all_landmarks list of that object
        '''
        for pose in data:
            count=0
            for lm in pose:
                self.all_3dlandmarks.append({"frame":frno,
                    "landmark":count,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility,
                    "presence": lm.presence
                })
                count+=1


        

    def list_into_csv(self,filename=None):
        '''copies the content from the all_landmarks list of the object and uploads it into a csv file.
        Name of the csv file is the input given to the function
        if no input then title of the object
        if no title then just "video"
        '''
        if not(filename):
            filename=self.name
        with open(f'{filename}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame','landmark', 'x', 'y', 'z', 'visibility', 'presence'])
            for lm in self.all_landmarks:
                writer.writerow([lm["frame"],lm["landmark"], lm["x"], lm["y"], lm["z"], lm["visibility"], lm["presence"]])
 
    def list3d_into_csv(self,filename=None):
        '''copies the content from the all_landmarks list of the object and uploads it into a csv file.
        Name of the csv file is the input given to the function
        if no input then title of the object
        if no title then just "video"
        '''
        if not(filename):
            filename=self.name
        with open(f'{filename}_3d.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame','landmark', 'x', 'y', 'z', 'visibility', 'presence'])
            for lm in self.all_3dlandmarks:
                writer.writerow([lm["frame"],lm["landmark"], lm["x"], lm["y"], lm["z"], lm["visibility"], lm["presence"]])
 
    

