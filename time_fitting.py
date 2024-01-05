import pandas as pd
import rtc_to_timenow
from datetime import datetime,timedelta
import cv2
import os
import numpy as np

class Time_fitting():
    '''
    time_fitting_polymate()またはtime_fitting()してからcut_video()
    最後まで使うならafter_time=-1
    '''
    def __init__(self,experiment_num:int,subject_num:int,before_time:float=0,after_time:float=-1,delay=0,path_to_data_dir=r"C:\Users\tyasu\Desktop\修士研究用",**kwargs):
        self.experiment_num = experiment_num
        self.delay=delay
        self.smarteye_df:pd.DataFrame=None
        self.polymate_df:pd.DataFrame=None
        self.DS_log_df:pd.DataFrame=None
        for key,value in kwargs.items():
            if key=="smarteye_df":
                self.smarteye_df = value
            elif key=="polymate_df":
                self.polymate_df = value
            elif key=="DS_log_df":
                self.DS_log_df = value
        self.time_fit_polymate_df:pd.DataFrame=None
        self.time_fit_smarteye_df:pd.DataFrame=None
        self.time_fit_DS_log_df:pd.DataFrame=None
        self.time_fit_polymate_dfs4emg:list[pd.DataFrame]=[]
        self.polymate_start_index=0
        self.polymate_end_index=len(self.polymate_df)-1
        self.smarteye_start_index=0
        self.smarteye_end_index=len(self.smarteye_df)-1
        self.DS_log_start_index=0
        self.DS_log_end_index=len(self.DS_log_df)-1
        self.path_to_video=os.path.join(path_to_data_dir,"実験データ",f"{subject_num}","camera",f"{experiment_num}.mp4")
        self.cut_video_output_folder=os.path.join(path_to_data_dir,"実験データ",f"{subject_num}","camera","cut")
        self.before_time=before_time
        self.after_time=after_time
        self.smarteye_time=None
        self.polymate_time=None
        self.DS_log_time:np.array=self.DS_log_df.values[:,11]
        self.path_to_data_dir=path_to_data_dir
        self.sampling_rate=2000#Hz
        self.before10s_df=None
        

    def rtc_to_real_time_all(self,delay:float=0)->list[datetime]:
        # RealTimeClock 列を抽出し、NumPy配列に変換
        real_time_clock_column = self.smarteye_df['RealTimeClock'].values
        times=[]
        for rtc in real_time_clock_column:
            times.append(rtc_to_timenow.calculate_real_time(rtc,delay))
        print(f"smarteye\nstart time:{times[0]},end time:{times[-1]}")
        return times
    
    def polymate_time_str_to_datetime(self,time_str:str):
        dt = datetime.strptime(time_str+"000", "%H:%M:%S.%f")
        return dt

    def time_of_polymate(self):

        clocks_str=self.polymate_df["CLOCK"].values
        times=[]
        for clock_str in clocks_str:
            dt = self.polymate_time_str_to_datetime(clock_str)
            times.append(dt)
        print(f"polymate\nstart time:{times[0]},end time:{times[-1]}")
        return times

    def seconds_to_polymate_index(self,polymate_time:list[datetime],second:float,base_time:datetime):
        return polymate_time.index(min(polymate_time,key=lambda x:abs(datetime.combine(datetime.today(),(base_time+timedelta(seconds=second)).time())-datetime.combine(datetime.today(),x.time())).total_seconds()))

    def calculate_polymate_index(self,smarteye_time,polymate_time):    
        start_time=smarteye_time[0]
        end_time=smarteye_time[-1]

        self.polymate_start_index=self.seconds_to_polymate_index(polymate_time,self.before_time,start_time)       
        print("polymate cut start:",self.polymate_start_index,polymate_time[self.polymate_start_index])

        if self.after_time==-1:
            self.polymate_end_index=self.seconds_to_polymate_index(polymate_time,0,end_time)
            print("polymate cut end:",self.polymate_end_index,polymate_time[self.polymate_end_index])
        else:
            self.polymate_end_index=self.seconds_to_polymate_index(polymate_time,self.after_time,start_time)
            print("polymate cut end:",self.polymate_end_index,polymate_time[self.polymate_end_index])
            
        return self.polymate_start_index, self.polymate_end_index

    def time_fitting_polymate(self,check_delay:bool=False):
        if self.smarteye_time==None:
            self.smarteye_time=self.rtc_to_real_time_all(self.delay)
        self.polymate_time=self.time_of_polymate()
        if check_delay:
            while True:
                user_input = input("正しければyを，正しくなければnを入力してください: ")
                if user_input.lower() == "y":
                    print("y が入力されました。")
                    break
                elif user_input.lower() == "n":
                    print("n が入力されました。")
                    raise SystemExit
                else:
                    print("無効な入力です。y か n を入力してください。")
                
        index=self.calculate_polymate_index(self.smarteye_time,self.polymate_time)
        self.time_fit_polymate_df=self.polymate_df.iloc[index[0]:index[1]+1].reset_index(drop=True)
        self.before10s_df=self.polymate_df.iloc[index[0]-self.sampling_rate*10:index[0]].reset_index(drop=True)
        # print("fit polymate dataframe",self.time_fit_polymate_df)
        return self.time_fit_polymate_df,self.before10s_df
    
    def seconds_to_smarteye_index(self,smarteye_time:list[datetime],second:float):
        start_time=smarteye_time[0]
        return smarteye_time.index(min(smarteye_time,key=lambda x:abs(datetime.combine(datetime.today(),(start_time+timedelta(seconds=second)).time())-datetime.combine(datetime.today(),x.time())).total_seconds()))
    
    def calculate_smarteye_index(self,smarteye_time):
        self.smarteye_start_index=self.seconds_to_smarteye_index(smarteye_time,self.before_time)
        print("smarteye cut start:",self.smarteye_start_index,smarteye_time[self.smarteye_start_index])
        
        if self.after_time==-1:
            self.smarteye_end_index=len(smarteye_time)-1
            print("smarteye non-cut end:",self.smarteye_end_index,smarteye_time[self.smarteye_end_index])
        else:
            self.smarteye_end_index=self.seconds_to_smarteye_index(smarteye_time,self.after_time)
            print("smarteye cut end:",self.smarteye_end_index,smarteye_time[self.smarteye_end_index])
        return self.smarteye_start_index, self.smarteye_end_index
        
    def time_fitting_smarteye(self,delete_during_stop:bool):
        if self.smarteye_time==None:
            self.smarteye_time=self.rtc_to_real_time_all(self.delay)
        index=self.calculate_smarteye_index(self.smarteye_time)
        time_fit_smarteye_df=self.smarteye_df
        if(delete_during_stop):
            from convert_from_DS_index import search_seconds_in_DS_log
            your_car_stopping_time_array=search_seconds_in_DS_log(DS_log_df=self.DS_log_df,search_for='your car stop')
            for subarray in your_car_stopping_time_array:
                start_index=self.seconds_to_smarteye_index(self.smarteye_time,subarray[0])
                end_index=self.seconds_to_smarteye_index(self.smarteye_time,subarray[1])+1
                if(start_index in time_fit_smarteye_df.index and end_index in time_fit_smarteye_df.index):
                    time_fit_smarteye_df=time_fit_smarteye_df.drop(range(start_index,end_index))
            time_fit_smarteye_df=time_fit_smarteye_df.loc[index[0]:index[1]+1]
            
        else:
            time_fit_smarteye_df=time_fit_smarteye_df.loc[index[0]:index[1]+1]
        self.time_fit_smarteye_df=time_fit_smarteye_df.reset_index(drop=True)
        print(self.time_fit_smarteye_df.shape)
        # print("fit smarteye dataframe",self.time_fit_smarteye_df)
        return self.time_fit_smarteye_df
    
    def calculate_DS_log_index(self,DS_log_time):
        start_time_difference_list=[]
        end_time_difference_list=[]
        DS_log_time=list(DS_log_time)

        for ds_time in DS_log_time:
            start_time_difference_list.append(abs(self.before_time-ds_time))
            end_time_difference_list.append(abs(self.after_time-ds_time))

        self.DS_log_start_index=DS_log_time.index(min(DS_log_time,key=lambda x:abs(self.before_time-x)))
        print("DS_log cut start:",self.DS_log_start_index,DS_log_time[self.DS_log_start_index])
        
        if self.after_time==-1:
            self.DS_log_end_index=len(DS_log_time)-1
            print("DS_log non-cut end:",self.DS_log_end_index,DS_log_time[self.DS_log_end_index])
        else:
            self.DS_log_end_index=DS_log_time.index(min(DS_log_time,key=lambda x:abs(self.after_time-x)))
            print("DS_log cut end:",self.DS_log_end_index,DS_log_time[self.DS_log_end_index])

        return self.DS_log_start_index, self.DS_log_end_index
    
    def time_fitting_DS_log(self):
        index=self.calculate_DS_log_index(self.DS_log_time)
        self.time_fit_DS_log_df=self.DS_log_df.iloc[index[0]:index[1]+1].reset_index(drop=True)
        # print("fit DS_log dataframe",self.time_fit_DS_log_df)
        return self.time_fit_DS_log_df
            
            
    def time_fitting(self,check_delay:bool=False):
        self.time_fitting_polymate()
        self.time_fitting_smarteye()
        self.time_fitting_DS_log()
        if check_delay:
            while True:
                user_input = input("正しければyを，正しくなければnを入力してください: ")
                if user_input.lower() == "y":
                    print("y が入力されました。")
                    break
                elif user_input.lower() == "n":
                    print("n が入力されました。")
                    raise SystemExit
                else:
                    print("無効な入力です。y か n を入力してください。")
        
    def polymate_clock_to_seconds_from_start(self,clock):
        # clockは%H:%M:%S.%fの形をした文字列
        time_difference=self.polymate_time_str_to_datetime(clock)-self.polymate_time_str_to_datetime(self.polymate_df["CLOCK"].values[0])
        seconds_difference=time_difference.total_seconds()
        return seconds_difference
    
    def cut_video(self,output_file_name):
        # 出力フォルダが存在しない場合、作成する
        if not os.path.exists(self.cut_video_output_folder):
            os.makedirs(self.cut_video_output_folder)

        # 出力動画ファイル名を指定
        output_file = os.path.join(self.cut_video_output_folder, output_file_name)
        # 入力動画を開く
        cap = cv2.VideoCapture(self.path_to_video)


        # 入力動画のプロパティを取得
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_rate = int(cap.get(5))
        # fourcc = int(cap.get(6))
        fourcc=cv2.VideoWriter_fourcc(*'XVID')
        start_time=self.polymate_clock_to_seconds_from_start(self.time_fit_polymate_df["CLOCK"].values[0])
        end_time=self.polymate_clock_to_seconds_from_start(self.time_fit_polymate_df["CLOCK"].values[-1])
        # 開始時間と終了時間をフレーム番号に変換
        start_frame = int(start_time * frame_rate)
        end_frame = int(end_time * frame_rate)

        # 出力動画を設定
        out = cv2.VideoWriter(output_file, fourcc, frame_rate, (frame_width, frame_height))

        # カットの範囲内のフレームを取得して出力
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # カット範囲外のフレームをスキップ
            if frame_count < start_frame:
                continue
            if frame_count > end_frame:
                break

            out.write(frame)

        # リソースの解放
        cap.release()
        out.release()

        print(f"finished cutting video:{output_file}")

        return output_file
    
    
    def time_fitting_polymate4emg(self,before_time:float=0,after_time:float=-1):
        if self.smarteye_time==None:
            self.smarteye_time=self.rtc_to_real_time_all(self.delay)
        if self.polymate_time==None:
            self.polymate_time=self.time_of_polymate()

        index=self.calculate_polymate_index4emg(self.smarteye_time,self.polymate_time,before_time,after_time)
        time_fitting_df=self.polymate_df.iloc[index[0]:index[1]+1].reset_index(drop=True)
        self.time_fit_polymate_dfs4emg.append(time_fitting_df)
        # print("fit polymate dataframe",self.time_fit_polymate_df)
        return time_fitting_df
    
    def calculate_polymate_index4emg(self,smarteye_time,polymate_time,before_time:float=0,after_time:float=-1):    
        start_time=smarteye_time[0]
        end_time=smarteye_time[-1]

        polymate_start_index=self.seconds_to_polymate_index(polymate_time,before_time,start_time)       
        print("polymate cut start:",polymate_start_index,polymate_time[polymate_start_index])

        if after_time==-1:
            polymate_end_index=self.seconds_to_polymate_index(polymate_time,0,end_time)
            print("polymate cut end:",polymate_end_index,polymate_time[polymate_end_index])
        else:
            polymate_end_index=self.seconds_to_polymate_index(polymate_time,after_time,start_time)
            print("polymate cut end:",polymate_end_index,polymate_time[polymate_end_index])
            
        return polymate_start_index, polymate_end_index


    