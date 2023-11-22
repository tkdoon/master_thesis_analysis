import pandas as pd
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import os

sampling_rate = 2000

def emg_analysis(emg_df:pd.DataFrame,experiment_num:int,subject_num:int,show:bool=True,store:bool=True):
    emg_np = emg_df.values
    emg_data = emg_np[:, 5].astype(np.float32)

    signals, info = nk.emg_process(
        emg_data, sampling_rate=sampling_rate, report="text", method_activation="mixture"
        # ,method_cleaning=None
        )#https://neuropsychology.github.io/NeuroKit/functions/emg.html
    print("info:",info)

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
            if not os.path.exists(fr"C:\Users\tyasu\Desktop\修士研究用\解析データ\{subject_num}"):
                os.makedirs(fr"C:\Users\tyasu\Desktop\修士研究用\解析データ\{subject_num}")
            if not os.path.exists(fr"C:\Users\tyasu\Desktop\修士研究用\解析データ\{subject_num}\EMG"):
                os.makedirs(fr"C:\Users\tyasu\Desktop\修士研究用\解析データ\{subject_num}\EMG")
            plt.savefig(fr"C:\Users\tyasu\Desktop\修士研究用\解析データ\{subject_num}\EMG\EMG_{experiment_num}.png")
        if show:
            plt.show()
        else:
            plt.close()

        print(nk.emg_intervalrelated(signals))
    else:
        print("筋運動なし")
    return {"signals":signals,"info":info,"times":times,"mean_time":mean_time}
        
        
        
        
if __name__=="__main__":
    path = r"C:\Users\tyasu\Desktop\修士研究用\実験データ\1\polymate\実験1.CSV"
    emg_df = pd.read_csv(path, header=3)
    emg_analysis(emg_df,1,1)