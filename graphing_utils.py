import plotly.express as px
import plotly.io as pio
import pandas as pd
from video_data import Video
pio.renderers.default = 'browser'

class GraphingUtils():

    def plot2d(self, video:Video,frame_id):
        """Plots the x and y landmarks of a given frame no. and video on the graph
        """
        v_width=9.6
        v_height=5.4
        frame_df = video.dframe[video.dframe['frame']==frame_id].copy()
        frame_df['x']*=9.6
        frame_df['y']*=5.4

        fig=px.scatter(frame_df,x='x',y='y',text='landmark',title=f"2D Pose Landmarks - {frame_id}"
                       ,range_x=(0,v_width),range_y=(0,v_height))

        fig.update_yaxes(autorange="reversed")
        fig.update_traces(textposition="top center")
        fig.show()

    def plot_landmark_with_time(self,video:Video,axis:str,ldmk):
        time_df = video.dframe[video.dframe['landmark']==ldmk].copy()
        time_df['frame']/=23
        time_df['x']*=9.6
        time_df['y']*=5.4

        fig = px.line(time_df, x="frame",y=axis,
                    title=f"2D_Data Pose Landmark Movement - Landmark {ldmk}",
                    labels={"frame":"Time"},
                    range_x=(0,12),markers=True)
        fig.show()
        

