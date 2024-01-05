import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from biosppy.signals import ecg
import os
import japanize_matplotlib

class ECG_analysis():
    def __init__(self,subject_num:int,experiment_num:int=None,experiment_nums:list=None,ecg_df:pd.DataFrame =None,before10s_df:pd.DataFrame=None,ecg_dfs:list[pd.DataFrame]=None,path_to_data_dir=r"C:\Users\tyasu\Desktop\修士研究用") -> None:
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
        self.outlier_num=None
        self.used_df_num=None
        self.outlier_rate=None
        self.before10s_rri=None
        self.normalized_sdnn=None
        self.normalized_max_rri=None
        self.normalized_min_rri=None
        self.normalized_mean_rri=None
        self.analysis_path=os.path.join(path_to_data_dir,"解析データ",str(subject_num),"ECG")
        if not os.path.exists(os.path.join(path_to_data_dir,"解析データ",str(subject_num))):
            os.makedirs(os.path.join(path_to_data_dir,"解析データ",str(subject_num)))
        if not os.path.exists(os.path.join(path_to_data_dir,self.analysis_path)):
            os.makedirs(os.path.join(path_to_data_dir,self.analysis_path))
        self.before10s_df=before10s_df

        
    def find_peaks(self,ecg_np):
        ecg_data = ecg.ecg(
            signal=ecg_np, sampling_rate=2000., show=False, interactive=False)

        (ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate) = ecg_data

        rpeaks_christov, = ecg.christov_segmenter(filtered, 2000.)
        # クリストフのアルゴリズムを使用
        ts_peaks = ts[rpeaks_christov]
        return ts_peaks,ts,filtered
    
    def box_hide_fix_rri(self,rri_data,ts_peaks_data):
        # 第1四分位数 (Q1) と第3四分位数 (Q3) を計算
        q1 = np.percentile(rri_data, 25)
        q3 = np.percentile(rri_data, 75)
        
        # IQR (四分位範囲) を計算
        iqr = q3 - q1
        
        # 外れ値の閾値を計算
        # 1.5だと外れ値以外も外れ値にしてしまう気がする
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 外れ値のインデックスを取得
        outliers = np.where((rri_data < lower_bound) | (rri_data > upper_bound))[0]
        fixed_rri_data = np.delete(rri_data, outliers)
        fixed_ts_peaks_data=np.delete(ts_peaks_data, outliers)
        return {"rri":fixed_rri_data, "ts_peaks":fixed_ts_peaks_data,"outlier_num":len(outliers)}
        
    
    def fix_rri(self,rri_data,ts_peaks_data):
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
        fixed_rri=_rri
        fixed_ts_peaks=_ts_peaks
        return {"rri":fixed_rri, "ts_peaks":fixed_ts_peaks}
    
    def easy_fix_rri(self,rri_data,ts_peaks_data):
        rri_mean=np.mean(rri_data)
        outliers=np.where(np.logical_or(rri_data < rri_mean*0.6, rri_data > rri_mean*1.4))[0]
        fixed_rri=np.delete(rri_data,outliers)
        fixed_ts_peaks=np.delete(ts_peaks_data,outliers)
        return {"rri":fixed_rri, "ts_peaks":fixed_ts_peaks,"outlier_num":len(outliers)}
    
    def calculate_peaks(self,ecg_np:np.array,fix:bool=True)->dict:
        ts_peaks,ts,filtered=self.find_peaks(ecg_np)
        rri = np.diff(ts_peaks) * 1000
        original_rri=rri
        
        outlier_num=None
        print("original rri:",original_rri)
        if fix:
            fixed_data=self.easy_fix_rri(rri,ts_peaks)
            rri=fixed_data["rri"]
            ts_peaks =fixed_data["ts_peaks"] 
            outlier_num=fixed_data["outlier_num"]
            
        return {"ts_peaks":ts_peaks,"rri":rri,"outlier_num":outlier_num,"original_rri":original_rri,"ts":ts,"filtered":filtered}
            
    def calculate_parameters(self,ts_peaks,rri,outlier_num,original_rri):
        self.outlier_num=outlier_num
        self.max_beat=1.0/(np.min(rri)/1000/60)#1分あたりの回数
        self.min_beat=1.0/(np.max(rri)/1000/60)
        self.mean_beat=1.0/(np.mean(rri)/1000/60)
        print(f"max:{self.max_beat}回/min\nmin:{self.min_beat}回/min\nmean:{self.mean_beat}回/min")                  
        print("fixed rri:", rri)
        spline_func = scipy.interpolate.interp1d(ts_peaks[:-1], rri, kind='cubic')
        ts_1sec = np.arange(ts_peaks[0], ts_peaks[-2], 1)
        rri_1sec = spline_func(ts_1sec).round(6)
        x = np.arange(ts_peaks[0], ts_peaks[-2], 0.05)
        y = spline_func(x)

        sdnn = np.std(rri)
        self.sdnn = sdnn
        # rriの標準偏差
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
        
        if(self.before10s_rri!=None):
            normalized_rri=rri/self.before10s_rri
            self.normalized_sdnn=np.std(normalized_rri)
            self.normalized_max_rri=np.max(normalized_rri)
            self.normalized_min_rri=np.min(normalized_rri)
            self.normalized_mean_rri=np.mean(normalized_rri)
        
        return {"x":x,"y":y,"ts_1sec":ts_1sec,"rri_1sec":rri_1sec,"frequencies":frequencies,"amp_fft_rri":amp_fft_rri}
    
        
    
    
    def calculate_before10s_mean_rate(self,before10s_np,fix:bool=True):
        ts_peaks,ts,filtered=self.find_peaks(before10s_np)
        rri = np.diff(ts_peaks) * 1000
        outlier_num=None
        if fix:
            fixed_data=self.easy_fix_rri(rri,ts_peaks)
            rri=fixed_data["rri"]
            ts_peaks =fixed_data["ts_peaks"] 
            outlier_num=fixed_data["outlier_num"]
        return np.mean(rri)
    
    def get_result(self,fix:bool=True,before10s_res=None):
        if(fix):
            if(before10s_res==None):
                before10s_res,before10s_index=self.choose_best_rri_data(self.before10s_df.values[:, 4]*(-1),self.before10s_df.values[:, 2],self.before10s_df.values[:, 3],self.before10s_df.values[:, 6],self.before10s_df.values[:, 7])
            self.before10s_rri=np.mean(before10s_res["rri"])
            res,index=self.choose_best_rri_data(self.ecg_df.values[:, 4]*(-1),self.ecg_df.values[:, 2],self.ecg_df.values[:, 3],self.ecg_df.values[:, 6],self.ecg_df.values[:, 7])
        else:
            if(before10s_res==None):
                before10s_res=self.calculate_peaks(self.ecg_df.values[:, 4]*(-1),fix)
            self.before10s_rri=np.mean(before10s_res["rri"])
            res_peaks=self.calculate_peaks(self.ecg_df.values[:, 4]*(-1),fix)
            res={**self.calculate_parameters(**res_peaks),**res_peaks}
        self.ts=res["ts"]
        self.filtered_data=res["filtered"]

        return res,before10s_res
    
    def analyze_single(self,show:bool=True,store:bool=True,font_size:int=12,fix:bool=True):
        res,before10s_res=self.get_result(fix=fix)
        self.show_single_all(res=res,show=show,store=store,font_size=font_size)
        return res,before10s_res
        
    
    def choose_best_rri_data(self,*data_list):
        """一番使いたいデータを可変引数の一番最初に持ってきてください．そうすると，外れ値の数が同じときに一番使いたいデータの計算結果を返します．"""
        res_list=[]
        min_outlier=None
        min_outliers_idx=0
        for i,data in enumerate(data_list):
            res=self.calculate_peaks(data,fix=True)
            res_list.append(res)
            if(min_outlier==None):
                min_outlier=res["outlier_num"]
                min_outliers_idx=i
            elif(res["outlier_num"]<min_outlier):
                min_outlier=res["outlier_num"]
                min_outliers_idx=i
            else:
                pass
            print(i,":",res["outlier_num"])
        print("min_outliers_idx:",min_outliers_idx,"\nmin_outlier:",min_outlier)
        self.outlier_num=min_outlier
        self.used_df_num=min_outliers_idx
        self.outlier_rate=min_outlier/len(res_list[min_outliers_idx]["original_rri"])
        used_res=res_list[min_outliers_idx]
        graph_elements=self.calculate_parameters(used_res["ts_peaks"],used_res["rri"],used_res["outlier_num"],used_res["original_rri"])
        
        return {**used_res,**graph_elements},min_outliers_idx
    
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
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"RRI",f"RRI_{self.experiment_num}.png"))
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
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"FFT_spectrum",f"FFT_{self.experiment_num}.png"))
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
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"heart_rate_rri",f"heartrate_rri_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
            
    def show_single_all(self,res,show:bool=True,store:bool=True,font_size:int=12):
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

            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"RRI",f"multi_RRI_{filename}.png"))
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

            
    def show_multiple(self,titles:list=[],large_title_rri:str=None,large_title_fft:str=None,large_title_mean_beat:str=None,show:bool=True,store:bool=True,font_size:int=12,fix:bool=True):
        n=len(self.ecg_dfs)
        res_list=[]
        for ecg_df in self.ecg_dfs:
            if(fix):
                res,index=self.choose_best_rri_data(ecg_df.values[:, 4]*(-1),ecg_df.values[:, 2],ecg_df.values[:, 3],ecg_df.values[:, 6],ecg_df.values[:, 7])
            else:
                res=self.calculate(ecg_df.values[:, 4]*(-1),fix)
            res_list.append(res)
        filename=""
        for experiment_num in self.experiment_nums:
            filename+=str(experiment_num)
        self.show_multiple_rri(res_list,n,filename,titles,large_title_rri,show,store,font_size)
        self.show_multiple_fft(res_list,n,filename,titles,large_title_fft,show,store,font_size)


        
        
            
    
        
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