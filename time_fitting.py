import pandas as pd
import rtc_to_timenow
from datetime import datetime,timedelta
import cv2
import os

class Time_fitting():
    '''
    time_fitting_polymate()してからcut_video()
    '''
    def __init__(self,experiment_num:int,subject_num:int,before_time:int=0,after_time:int=-1,delay=0,path_to_data_dir=r"C:\Users\tyasu\Desktop\修士研究用"):
        self.experiment_num = experiment_num
        self.delay=delay
        self.path_to_polymate_log = os.path.join(path_to_data_dir,"実験データ",f"{subject_num}","polymate","new",f"実験{experiment_num}.CSV")
        self.path_to_smarteye_log=os.path.join(path_to_data_dir,"実験データ",f"{subject_num}","smarteye",f"{experiment_num}.log")
        self.path_to_DS_log=os.path.join(path_to_data_dir,"実験データ",f"{subject_num}","ds_log",f"ds{experiment_num}.csv")
        self.smarteye_df:pd.DataFrame=pd.read_csv(self.path_to_smarteye_log, sep='\t')  # 列名はタブ文字で区切られている
        self.polymate_df:pd.DataFrame=pd.read_csv(self.path_to_polymate_log, header=3)
        self.DS_log_df:pd.DataFrame=pd.read_csv(self.path_to_DS_log, header=6,encoding="shift-jis")
        self.time_fit_polymate_df:pd.DataFrame=None
        self.time_fit_smarteye_df:pd.DataFrame=None
        self.time_fit_DS_log_df:pd.DataFrame=None
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
        self.DS_log_time=self.DS_log_df.values[:,11]
        self.path_to_data_dir=path_to_data_dir
        

    def rtc_to_real_time_all(self,delay:float=0):
        # RealTimeClock 列を抽出し、NumPy配列に変換
        real_time_clock_column = self.smarteye_df['RealTimeClock'].values
        times=[]
        for rtc in real_time_clock_column:
            times.append(rtc_to_timenow.calculate_real_time(rtc,delay))
        print(f"smarteye\nstart time:{times[0]},end time:{times[-1]}")
        return times
    
    def polymate_time_str_to_datetime(self,time_str):
        dt = datetime.strptime(time_str, "%H:%M:%S.%f")
        return dt

    def time_of_polymate(self):

        clocks_str=self.polymate_df["CLOCK"].values
        times=[]
        for clock_str in clocks_str:
            dt = self.polymate_time_str_to_datetime(clock_str)
            times.append(dt)
        print(f"polymate\nstart time:{times[0]},end time:{times[-1]}")
        return times


    def calculate_polymate_index(self,smarteye_time,polymate_time):    
        start_time=smarteye_time[0]
        end_time=smarteye_time[-1]
        for index,p_time in enumerate(polymate_time):
            if((start_time.replace(microsecond=0)+timedelta(seconds=self.before_time)).time()==p_time.replace(microsecond=0).time()):#シナリオが始まってから使う部分に到達したら
                print("polymate cut start:",index,p_time)
                self.polymate_start_index=index
                break
        for index,p_time in enumerate(reversed(polymate_time)):
            if self.after_time==-1:
                if(end_time.replace(microsecond=0).time()==p_time.replace(microsecond=0).time()):
                    print("polymate cut end:",len(polymate_time)-index,p_time)
                    self.polymate_end_index=len(polymate_time)-index
                    break
            else:
                if((start_time.replace(microsecond=0)+timedelta(seconds=self.before_time)+timedelta(seconds=self.after_time)).time()==p_time.replace(microsecond=0).time()):#シナリオが始まってから使う部分が終了したら
                    print("polymate cut end:",len(polymate_time)-index,p_time)
                    self.polymate_end_index=len(polymate_time)-index
                    break
        return self.polymate_start_index, self.polymate_end_index

    def time_fitting_polymate(self,check_delay:bool=False):
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
        self.time_fit_polymate_df=self.polymate_df.iloc[index[0]:index[1]].reset_index(drop=True)
        # print("fit polymate dataframe",self.time_fit_polymate_df)
        return self.time_fit_polymate_df
    
    def polymate_clock_to_seconds_from_start(self,clock):
        # clockは%H:%M:%S.%fの形をした文字列
        time_difference=self.polymate_time_str_to_datetime(clock)-self.polymate_time_str_to_datetime(self.polymate_df["CLOCK"].values[0])
        seconds_difference=time_difference.total_seconds()
        return seconds_difference
    
    def calculate_smarteye_index(self,smarteye_time):
        start_time=smarteye_time[0]
        end_time=smarteye_time[-1]
        for index,s_e_time in enumerate(smarteye_time):
            if((start_time.replace(microsecond=0)+timedelta(seconds=self.before_time)).time()==s_e_time.replace(microsecond=0).time()):#シナリオが始まってから使う部分に到達したら
                print("smarteye cut start:",index,s_e_time)
                self.smarteye_start_index=index
                break
        if self.after_time==-1:
            print("smarteye non-cut end:",len(smarteye_time)-1,s_e_time)
            self.smarteye_end_index=len(smarteye_time)-1
        else:
            for index,s_e_time in enumerate(reversed(smarteye_time)):
                    if((start_time.replace(microsecond=0)+timedelta(seconds=self.before_time)+timedelta(seconds=self.after_time)).time()==s_e_time.replace(microsecond=0).time()):#シナリオが始まってから使う部分が終了したら
                        print("smarteye cut end:",len(smarteye_time)-index,s_e_time)
                        self.smarteye_end_index=len(smarteye_time)-index
                        break
        return self.smarteye_start_index, self.smarteye_end_index
        
        
    def time_fitting_smarteye(self):
        if self.smarteye_time==None:
            self.smarteye_time=self.rtc_to_real_time_all(self.delay)
        index=self.calculate_smarteye_index(self.smarteye_time)
        self.time_fit_smarteye_df=self.smarteye_df.iloc[index[0]:index[1]].reset_index(drop=True)
        # print("fit smarteye dataframe",self.time_fit_smarteye_df)
        return self.time_fit_smarteye_df
    
    def calculate_DS_log_index(self,DS_log_time):
        for index,DS_time in enumerate(DS_log_time):
            if(DS_time==self.before_time):
                self.DS_log_start_index=index
                print("DS_log cut start:",index,DS_time)
                break
        if self.after_time==-1:
            print("DS_log non-cut end:",len(DS_log_time)-1,DS_time)
            self.DS_log_end_index=len(DS_log_time)-1
        else:
            for index,DS_time in enumerate(reversed(DS_log_time)):
                if(DS_time==self.before_time+self.after_time):#シナリオが始まってから使う部分が終了したら
                    print("DS_log cut end:",len(DS_log_time)-index,DS_time)
                    self.DS_log_end_index=len(DS_log_time)-index
                    break
        return self.DS_log_start_index, self.DS_log_end_index
    
    
    def time_fitting_DS_log(self):
        index=self.calculate_DS_log_index(self.DS_log_time.astype(int))
        self.time_fit_DS_log_df=self.DS_log_df.iloc[index[0]:index[1]].reset_index(drop=True)
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
        
            
    def cut_video(self):
        output_file_name = f'cut{self.experiment_num}.mp4'
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
        fourcc = int(cap.get(6))
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