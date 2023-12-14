import pandas as pd
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import os
from biosppy.signals import emg,tools
import scipy 
from PIL import Image


sampling_rate=2000
def emg_analysis(emg_df:pd.DataFrame,experiment_num:int,subject_num:int,show:bool=True,store:bool=True,font_size:int=12,path_to_data_dir=r"C:\Users\tyasu\Desktop\修士研究用"):
    emg_np = emg_df.values
    emg_data = emg_np[:, 5].astype(np.float32)
    emg_data=scipy.signal.detrend(emg_data)
    # emg_data=delete_outlier_by_std(emg_data,5)
    # emg_data,_,_=tools.filter_signal(emg_data,sampling_rate=2000,ftype='butter',
    #                                   band='bandpass',
    #                                   order=4,
    #                                   frequency=(3,500)
    #                                  )

    signals, info = nk.emg_process(
        emg_data, sampling_rate=sampling_rate, report="text", method_activation="mixture"
        ,threshold=0.8
        # ,size=20
        # ,threshold_size=22
        ,method_cleaning=None
        )#https://neuropsychology.github.io/NeuroKit/functions/emg.html
    print("info:",info)
    # print(signals)

    signals_np = signals.values.astype(np.float32)
    times = []
    for index, frame in enumerate(signals_np):
        if(frame[4] == 1):
            onset = index
        if(frame[5] == 1):
            offset = index
            time = (offset-onset)/sampling_rate
            times.append(time)
            print(f"筋運動時間は{time}秒")

    mean_time = np.average(times)
    print(f"平均筋運動時間は{mean_time}秒,回数は{len(times)}回，総時間は{sum(times)}秒")

    if(len(info["EMG_Onsets"])!=0):
        nk.emg_plot(signals, info)
        # グラフを表示する
        fig = plt.gcf()
        fig.set_size_inches(16, 12, forward=True)

        if store:
            if not os.path.exists(os.path.join(path_to_data_dir,"解析データ",str(subject_num))):
                os.makedirs(os.path.join(path_to_data_dir,"解析データ",str(subject_num)))
            if not os.path.exists(os.path.join(path_to_data_dir,"解析データ",str(subject_num),"EMG")):
                os.makedirs(os.path.join(path_to_data_dir,"解析データ",str(subject_num),"EMG"))
            plt.savefig(os.path.join(path_to_data_dir,"解析データ",str(subject_num),"EMG",f"EMG_{experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()

        print(nk.emg_intervalrelated(signals))
    else:
        # fig = plt.figure(figsize=(16,12))
        # plt.rcParams["font.size"] = 16
        # plt.plot(frequencies,amp_fft_rri)
        # plt.title('FFT Amplitude Spectrum')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Amplitude')
        # plt.grid()
        # plt.xlim(0, 1)
        # plt.ylim(0, 50000)
        
        time_array=np.arange(len(emg_data))/sampling_rate
        fig=plt.figure(figsize=(64,8),dpi=300)
        plt.rcParams["font.size"] = font_size
        plt.plot(time_array,emg_data)
        plt.title('emg')
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')
        plt.ylim(top=80,bottom=-80)

        if store:
            if not os.path.exists(os.path.join(path_to_data_dir,"解析データ",str(subject_num))):
                os.makedirs(os.path.join(path_to_data_dir,"解析データ",str(subject_num)))
            if not os.path.exists(os.path.join(path_to_data_dir,"解析データ",str(subject_num),"EMG")):
                os.makedirs(os.path.join(path_to_data_dir,"解析データ",str(subject_num),"EMG"))
            plt.savefig(os.path.join(path_to_data_dir,"解析データ",str(subject_num),"EMG",f"EMG_{experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
        print("筋運動なし")
    return {"signals":signals,"info":info,"times":times,"mean_time":mean_time}


def median1d(arr, k):
    w = len(arr)
    idx = np.fromfunction(lambda i, j: i + j, (k, w), dtype=np.int) - k // 2
    idx[idx < 0] = 0
    idx[idx > w - 1] = w - 1
    return np.median(arr[idx], axis=0)


def delete_outlier_by_std(data,n):
    std_value=np.std(data)
    data[data>std_value*n]=0
    data[data<-std_value*n]=0
    return data
       


def show_emg_graph(emg_df,subject_num,experiment_num,path_to_data_dir=r"C:\Users\tyasu\Desktop\修士研究用",font_size=16,store=True,show=True,figsize=(64,20),dpi:int=300):   
    analysis_path=os.path.join("解析データ",str(subject_num),"EMG")
    sampling_rate = 2000
    emg_np = emg_df.values
    emg_data = emg_np[:, 5].astype(np.float32)
    emg_data=scipy.signal.detrend(emg_data)
    # emg_data,_,_=tools.filter_signal(emg_data,sampling_rate=2000,ftype='butter',
    #                                 band='highpass',
    #                                 order=4,
    #                                 frequency=5
    #                                 )
    # emg_data=median1d(emg_data,91)
    time_array=np.arange(len(emg_data))/sampling_rate
    def make1graph(data,n,index,titles:list=[]):
        plt.subplot(n,1,index+1)
        plt.plot(time_array, data)
        plt.xlabel('time(s)')
        plt.ylabel('Amplitude')
        plt.grid()
    plt.figure(figsize=figsize)
    # plt.rcParams["figure.figsize"]=figsize
    plt.rcParams['figure.figsize'] = figsize   
    plt.rcParams["font.size"] = font_size
    plt.rcParams['figure.dpi'] = dpi  # 解像度を指定
    for index,data in enumerate(emg_np[:,2:8].T):
        make1graph(data,6,index)

    # plt.tight_layout()
    
    # plt.plot(time_array,emg_data)
    # plt.title('emg')
    # plt.xlabel('time (s)')
    # plt.ylabel('amplitude')
    # plt.ylim(top=80,bottom=-80)
    if store:
        if not os.path.exists(os.path.join(path_to_data_dir,analysis_path)):
            os.makedirs(os.path.join(path_to_data_dir,analysis_path))
        plt.savefig(os.path.join(path_to_data_dir,analysis_path,f"emg_graph_{experiment_num}.png"))
    if show:
        plt.show()
    else:
        plt.close()

def check_burst(emg_df,subject_num,experiment_num,path_to_data_dir=r"C:\Users\tyasu\Desktop\修士研究用")->list[tuple[float]]:
    sampling_rate = 2000
    emg_np = emg_df.values
    emg_data = emg_np[:, 5].astype(np.float32)
    emg_data=scipy.signal.detrend(emg_data)
    arv_data=np.abs(emg_data)
    check_list=detect_burst(arv_data,sampling_rate,0.05)  
    for i,check in enumerate(check_list):
        print(i,":",check[0],check[1])  
    show_emg_graph(emg_df,subject_num,experiment_num,store=True,show=False,figsize=(64,20),dpi=300,font_size=12,path_to_data_dir=path_to_data_dir)
    filename = os.path.join(path_to_data_dir,"解析データ",str(subject_num),"EMG",f"emg_graph_{experiment_num}.png")
    imgPIL = Image.open(filename)  # 画像読み込み
    imgPIL.show()  # 画像表示

    burst_list=[]
    if(len(check_list)!=0):
        while True:
            check_reply=list(input().split())
            if(len(check_reply)==len(check_list) and all(element == "n" or element == "y" for element in check_reply)):
                break
        for j,reply in enumerate(check_reply):
            if reply=="n":
                pass
            else:
                burst_list.append(check_list[j])
    return burst_list
        

def detect_burst(emg_data,sampling_rate,burst_time):
    """burst_timeの単位はsec"""
    sd=np.std(emg_data)
    mean=np.mean(emg_data)
    print(f"mean:{mean},sd:{sd}")
    threshold=mean+sd*2
    continuous_num=burst_time*sampling_rate
    indices = np.where(np.logical_or(emg_data > threshold,emg_data<-threshold))[0]
    def count_consecutive_elements(arr):
        count = 1
        count_list=[]
        for i in range(1, len(arr)):
            if arr[i] == arr[i - 1]+1:
                count += 1
            else:
                count_list.append((count,arr[i-count+1],arr[i]))
                count = 1

        return count_list

    count_list=count_consecutive_elements(indices)
    check_list=[]
    for count_info in count_list:
        if(count_info[0]>=continuous_num):
            check_list.append((count_info[1]/sampling_rate,count_info[2]/sampling_rate))
    return check_list
        
    
        
        
if __name__=="__main__":
    path = r"..\実験データ\1\polymate\実験1.CSV"
    emg_df = pd.read_csv(path, header=3)
    # emg_analysis(emg_df,1,1)
    # emg_np = emg_df.values
    # emg_data = emg_np[:, 5].astype(np.float32)
    # emg_data,_,_=tools.filter_signal(emg_data,sampling_rate=2000,ftype='butter',
    #                                   band='bandpass',
    #                                   order=4,
    #                                   frequency=(5,500)
    #                                  )
    # emg.silva_onset_detector(emg_data,sampling_rate=2000,size=60)
    burst_list=check_burst(emg_df,1,1)