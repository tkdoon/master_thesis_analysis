import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from biosppy.signals import ecg
import os
import japanize_matplotlib

class ECG_analysis():
    def __init__(self,subject_num:int,experiment_num:int=None,experiment_nums:list=None,ecg_df:pd.DataFrame =None,ecg_dfs:list=None,path_to_data_dir=r"C:\Users\tyasu\Desktop\修士研究用") -> None:
        self.ecg_df= ecg_df
        self.experiment_num = experiment_num
        self.ecg_dfs=ecg_dfs
        self.experiment_nums=experiment_nums
        self.subject_num = subject_num
        self.max_beat=None
        self.min_beat=None
        self.mean_beat=None
        self.hf=None
        self.lf=None
        self.hf_lf_rate=None
        self.path_to_data_dir=path_to_data_dir
        self.filtered_data=None
        self.ts=None
        self.sdnn=None
        self.analysis_path=os.path.join(path_to_data_dir,"解析データ",str(subject_num),"ECG")
        if not os.path.exists(os.path.join(path_to_data_dir,"解析データ",str(subject_num))):
            os.makedirs(os.path.join(path_to_data_dir,"解析データ",str(subject_num)))
        if not os.path.exists(os.path.join(path_to_data_dir,self.analysis_path)):
            os.makedirs(os.path.join(path_to_data_dir,self.analysis_path))
    
    def calculate(self,ecg_np:np.array,modify:bool=True)->dict:
        def find_peaks():
            ecg_data = ecg.ecg(
                signal=ecg_np, sampling_rate=2000., show=False, interactive=False)

            (ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate) = ecg_data
            self.filtered_data=filtered
            self.ts=ts
            rpeaks_christov, = ecg.christov_segmenter(filtered, 2000.)
            # クリストフのアルゴリズムを使用
            ts_peaks = ts[rpeaks_christov]
            return ts_peaks
        ts_peaks=find_peaks()
        rri = np.diff(ts_peaks) * 1000
        def modify_rri(rri_data,ts_peaks_data):
            _rri=rri_data
            _ts_peaks=ts_peaks_data
            while True:
                rri_mean=np.mean(_rri)
                # まず、指定の条件に合うデータを見つけて delete_index に格納
                condition = np.logical_or(_rri < rri_mean * 0.6, _rri > rri_mean * 1.4)
                delete_candidate_index = np.where(condition)
                if len(delete_candidate_index[0])==0:
                    break
                # 条件に合うデータのみを抽出
                filtered_data = _rri[condition]

                # 抽出したデータの中で rri_data - rri_mean の絶対値が最大のものを見つける
                max_abs_diff_index = np.argmax(np.abs(filtered_data - rri_mean))

                # 最大絶対値の差を持つデータのインデックスを取得
                delete_index = delete_candidate_index[0][max_abs_diff_index]
                _rri=np.delete(_rri,delete_index)
                _ts_peaks=np.delete(_ts_peaks,delete_index)
            modified_rri=_rri
            modified_ts_peaks=_ts_peaks
            return {"rri":modified_rri, "ts_peaks":modified_ts_peaks}
        def easy_modify_rri(rri_data,ts_peaks_data):
            delete_index=np.where(np.logical_or(rri_data < 650, rri_data > 1000))
            modified_rri=np.delete(rri_data,delete_index)
            modified_ts_peaks=np.delete(ts_peaks_data,delete_index)
            return {"rri":modified_rri, "ts_peaks":modified_ts_peaks}
        
        if modify:
            modified_data=modify_rri(rri,ts_peaks)
            rri=modified_data["rri"]
            ts_peaks =modified_data["ts_peaks"] 
        self.max_beat=1.0/(np.min(rri)/1000/60)#1分あたりの回数
        self.min_beat=1.0/(np.max(rri)/1000/60)
        self.mean_beat=1.0/(np.mean(rri)/1000/60)
        print(f"max:{self.max_beat}回/min\nmin:{self.min_beat}回/min\nmean:{self.mean_beat}回/min")                  
        print("rri:", rri)
        spline_func = scipy.interpolate.interp1d(ts_peaks[:-1], rri, kind='cubic')
        ts_1sec = np.arange(ts_peaks[0], ts_peaks[-2], 1)
        rri_1sec = spline_func(ts_1sec).round(6)
        x = np.arange(ts_peaks[0], ts_peaks[-2], 0.05)
        y = spline_func(x)

        sdnn = np.std(rri_1sec)
        self.sdnn = sdnn
        # rriの分散
        print("sdnn:", sdnn)

        fft_rri = np.fft.fft(y)
        amp_fft_rri = np.abs(fft_rri)
        N = len(y)  # データの点数
        sample_rate = 20  # サンプリングレート（適宜設定）,0.05の逆数かな？
        frequencies = np.fft.fftfreq(N, 1.0 / sample_rate)
        
        # 指定した周波数範囲のインデックスを見つける
        lf_low_index = np.where(frequencies >= 0.04)[0][0]
        lf_high_index = np.where(frequencies <= 0.15)[0][-1]

        # 振幅スペクトルから指定した範囲の振幅を取得
        lf_amplitudes_in_range = amp_fft_rri[lf_low_index:lf_high_index+1]

        # 振幅の合計を計算
        self.lf = np.sum(lf_amplitudes_in_range)

        print("lf:", self.lf)

        # 指定した周波数範囲のインデックスを見つける
        hf_low_index = np.where(frequencies >= 0.15)[0][0]
        hf_high_index = np.where(frequencies <= 0.4)[0][-1]

        # 振幅スペクトルから指定した範囲の振幅を取得
        hf_amplitudes_in_range = amp_fft_rri[hf_low_index:hf_high_index+1]

        # 振幅の合計を計算
        self.hf = np.sum(hf_amplitudes_in_range)
        self.hf_lf_rate=self.hf/self.lf
        print("hf:", self.hf)
        print("lf/hf:", self.hf_lf_rate)
        return {"ts_peaks":ts_peaks,"rri":rri,"x":x,"y":y,"ts_1sec":ts_1sec,"rri_1sec":rri_1sec,"frequencies":frequencies,"amp_fft_rri":amp_fft_rri,"mean_beat":self.mean_beat}
    
    def show_single_rri(self,ts_peaks,rri,ts_1sec,rri_1sec,x,y,show:bool=True,store:bool=True,font_size:int=12):
         # グラフを描画
        fig = plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        plt.scatter(ts_peaks[:-1], rri, marker='x', s=15, label='RRI', zorder=2)
        plt.scatter(ts_1sec, rri_1sec, s=15, color='darkorange',
                    label='RRI (interpolated)')
        plt.plot(x, y, color='darkorange', lw=1, zorder=-1)

        plt.xlabel('time [s]')
        plt.ylabel('RRI [ms]')
        plt.grid()
        plt.legend()

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"RRI")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"RRI"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"RRI","RRI_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
    def show_single_fft(self,frequencies,amp_fft_rri,show:bool=True,store:bool=True,font_size:int=12):
        fig = plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        plt.plot(frequencies,amp_fft_rri)
        plt.title('FFT Amplitude Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.xlim(0, 1)
        plt.ylim(0, 50000)

        if store:

            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"FFT_spectrum")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"FFT_spectrum"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"FFT_spectrum","FFT_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
    def show_beat_wave(self,ts_peaks,rri,ts_1sec,rri_1sec,x,y,show:bool=True,store:bool=True,font_size:int=12):
        fig = plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        plt.subplot(2,1,1)
        plt.plot(self.ts,self.filtered_data)
        plt.title('wave data')
        plt.xlabel('time(s)')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.subplot(2,1,2)
        plt.scatter(ts_peaks[:-1], rri, marker='x', s=15, label='RRI', zorder=2)
        plt.scatter(ts_1sec, rri_1sec, s=15, color='darkorange',
                label='RRI (interpolated)')
        plt.plot(x, y, color='darkorange', lw=1, zorder=-1)
        plt.title("RRI")
        plt.xlabel('time [s]')
        plt.ylabel('RRI [ms]')
        plt.grid()
        plt.legend()

        if store:

            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"heart_rate_rri")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"heart_rate_rri"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"heart_rate_rri","heartrate_rri_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
        
        
    def show_single(self,show:bool=True,store:bool=True,font_size:int=12,modify:bool=True):
        res=self.calculate(self.ecg_df.values[:, 4]*(-1),modify)
        ts_peaks=res["ts_peaks"]
        rri=res["rri"]
        x=res["x"]
        y=res["y"]
        ts_1sec=res["ts_1sec"]
        rri_1sec=res["rri_1sec"]
        frequencies=res["frequencies"]
        amp_fft_rri=res["amp_fft_rri"]
        self.show_single_rri(ts_peaks,rri,ts_1sec,rri_1sec,x,y,show,store,font_size)
        self.show_single_fft(frequencies,amp_fft_rri,show,store,font_size)
        self.show_beat_wave(ts_peaks,rri,ts_1sec,rri_1sec,x,y,show,store,font_size)
        
    def show_multiple_rri(self,calculate_res_list,n,filename,titles:list=[],large_title:str=None,show:bool=True,store:bool=True,font_size:int=12):
        def make1graph(ts_peaks,rri,ts_1sec,rri_1sec,x,y,n,index,titles:list=[]):
            plt.subplot(n,1,index+1)
            plt.scatter(ts_peaks[:-1], rri, marker='x', s=15, label='RRI', zorder=2)
            plt.scatter(ts_1sec, rri_1sec, s=15, color='darkorange',
                    label='RRI (interpolated)')
            plt.plot(x, y, color='darkorange', lw=1, zorder=-1)

            plt.xlabel('time [s]')
            plt.ylabel('RRI [ms]')
            plt.grid()
            plt.legend()
            if len(titles)!=0:
                plt.title(titles[index])
        fig = plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        for index,res in enumerate(calculate_res_list):
            ts_peaks=res["ts_peaks"]
            rri=res["rri"]
            x=res["x"]
            y=res["y"]
            ts_1sec=res["ts_1sec"]
            rri_1sec=res["rri_1sec"]
            make1graph(ts_peaks,rri,ts_1sec,rri_1sec,x,y,n,index,titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()

        if store:

            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"RRI","multi_RRI_{filename}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
    def show_multiple_fft(self,calculate_res_list,n,filename,titles:list=[],large_title:str=None,show:bool=True,store:bool=True,font_size:int=12)->None:
        def make1graph(frequencies,amp_fft_rri,n,index,titles:list=[]):
            plt.subplot(n,1,index+1)
            plt.plot(frequencies, amp_fft_rri)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.xlim(0, 1)
            plt.ylim(0, 50000)
            if len(titles)!=0:
                plt.title(titles[index])
            
        fig = plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        for index,res in enumerate(calculate_res_list):
            frequencies =res["frequencies"]
            amp_fft_rri=res["amp_fft_rri"]
            make1graph(frequencies,amp_fft_rri,n,index,titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()

        if store:

            plt.close()
            
            
    def show_multi_mean_beat(self,calculate_res_list,filename,titles:list[str],large_title:str,show:bool=True,store:bool=True,font_size:int=12):
        data1=[]
        for res in calculate_res_list:
            data1.append(res["mean_beat"])
                # サンプルデータ
        categories = titles


        # グラフの幅
        bar_width = 0.35
        
        if(len(categories)==0):
            categories=self.experiment_nums

        # インデックス
        indices = np.arange(len(categories))

        # データ1の棒グラフ
        plt.bar(indices, data1, bar_width, label='平均の心拍数')
        plt.ylim(bottom=50)
        plt.ylabel('heart rate(/min)')

        # カテゴリ名を設定
        plt.xticks(indices + bar_width / 2, categories)

        # 凡例を表示
        plt.legend()
        if(large_title):
        # グラフのタイトルと軸ラベル
            plt.title(large_title)
        plt.tight_layout()


        if store:

            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"mean_heart_rate")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"mean_heart_rate"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"mean_heart_rate","heart_rate_{filename}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
            
    def show_multiple(self,titles:list=[],large_title_rri:str=None,large_title_fft:str=None,large_title_mean_beat:str=None,show:bool=True,store:bool=True,font_size:int=12,modify:bool=True):
        n=len(self.ecg_dfs)
        res_list=[]
        for ecg_df in self.ecg_dfs:
            res=self.calculate(ecg_df.values[:, 4]*(-1),modify)
            res_list.append(res)
        filename=""
        for experiment_num in self.experiment_nums:
            filename+=str(experiment_num)
        self.show_multiple_rri(res_list,n,filename,titles,large_title_rri,show,store,font_size)
        self.show_multiple_fft(res_list,n,filename,titles,large_title_fft,show,store,font_size)
        self.show_multi_mean_beat(res_list,filename,titles,large_title_mean_beat,show,store,font_size)



        

        
            
    
        
    # def ecg_analysis(show:bool=True,store:bool=True,font_size:int=12):    



        # fft_result = np.fft.fft(amp_data)

        # amplitude = np.abs(fft_result)

        # N = len(amp_data)  # データの点数
        # sample_rate = 2000  # サンプリングレート（適宜設定）
        # frequencies = np.fft.fftfreq(N, 1.0 / sample_rate)

        # plt.figure(figsize=(8, 4))
        # plt.plot(frequencies, amplitude)
        # plt.title('FFT Amplitude Spectrum')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Amplitude')
        # plt.grid()
        # plt.show()
        # return {"rri":rri,"sdnn":sdnn,"lf":lf,"hf":hf,"lf/hf":lf/hf}


    # ecg_analysis(ecg_df)