import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import os


class Smarteye_analysis():
    def __init__(self,subject_num:int,experiment_num:int=None,experiment_nums:list[int]=None,smarteye_df:pd.DataFrame=None,smarteye_dfs:list[pd.DataFrame]=None) -> None:
        self.subject_num=subject_num
        self.experiment_num=experiment_num
        self.experiment_nums=experiment_nums
        self.smarteye_df=smarteye_df
        self.sampling_frequency=60#Hz
        if(smarteye_df is not None):
            self.smarteye_np=smarteye_df.values
            self.time_array=np.divide(self.smarteye_np[:,0].astype(int)-int(self.smarteye_np[0,0]),self.sampling_frequency)
        self.smarteye_dfs=smarteye_dfs
        
    def fix_data(self,np_data:np.array)->np.array:
        # 直前の値に置き換える
        # データの値が0でないインデックスを取得
        nonzero_indices = np.where(np_data != 0)[0]

        # データの値が0でない部分の平均値を計算
        mean_value = np.mean(np_data[nonzero_indices])
        
        prev_value = None

        for i in range(len(np_data)):
            if np_data[i] == 0:
                if prev_value is not None:
                    np_data[i] = prev_value
                else:
                    # 0の前に値がない場合、そのままにしておくか、別の初期値を設定することもできます。
                    np_data[i] = 0  # 0の前に値がない場合、0のままにする例

            else:
                prev_value = np_data[i]

        return np_data
        

        # データの値が0の部分を削除し、平均値で置き換える
        np_data[np_data == 0] = mean_value
        
        return np_data
        
        
        
    
    def show_gaze_direction(self,show:bool=True,store:bool=True):
        gaze_direction_array=self.smarteye_np[:,73:77]
        fig=plt.figure(figsize=(16,12))
        plt.plot(self.time_array,self.fix_data(gaze_direction_array[:,0]),label="direction-x")
        plt.plot(self.time_array,self.fix_data(gaze_direction_array[:,1]),label="direction-y")
        plt.plot(self.time_array,self.fix_data(gaze_direction_array[:,2]),label="direction-z")
        plt.plot(self.time_array,self.fix_data(gaze_direction_array[:,3]),label="direction-Q")
        plt.title('Gaze direction')
        plt.xlabel('time (s)')
        plt.ylabel('gaze direction')
        plt.legend()
        if store:
            if not os.path.exists(fr"C:\Users\tyasu\Desktop\修士研究用\解析データ\{self.subject_num}\smarteye\gaze_direction"):
                os.makedirs(fr"C:\Users\tyasu\Desktop\修士研究用\解析データ\{self.subject_num}\smarteye\gaze_direction")
            plt.savefig(fr"C:\Users\tyasu\Desktop\修士研究用\解析データ\{self.subject_num}\smarteye\gaze_direction\gaze_direction_{self.experiment_num}.png")
        if show:
            plt.show()
        else:
            plt.close()
            
    def show_multi_gaze_direction(self,filename,n,titles:list=[],large_title:str=None,store:bool=True,show:bool=True):
        def make1graph(time_array,gaze_direction_array,n,index,titles:list=[]):
            plt.subplot(n,1,index+1)
            plt.plot(time_array, self.fix_data(gaze_direction_array[:,0]),label="direction-x")
            plt.plot(time_array, self.fix_data(gaze_direction_array[:,1]),label="direction-y")
            plt.plot(time_array, self.fix_data(gaze_direction_array[:,2]),label="direction-z")
            plt.plot(time_array, self.fix_data(gaze_direction_array[:,3]),label="direction-Q")
            plt.xlabel('time(s)')
            plt.ylabel('gaze direction')
            plt.legend()
            plt.grid()

            if len(titles)!=0:
                plt.title(titles[index])
            
        fig = plt.figure(figsize=(16,12))
        for index,smarteye_log in enumerate(self.smarteye_dfs):
            smarteye_np=smarteye_log.values
            make1graph(np.divide(smarteye_np[:,0].astype(int)-int(smarteye_np[0,0]),self.sampling_frequency),smarteye_np[:,73:77],n,index,titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()
        if store:
            if not os.path.exists(fr"C:\Users\tyasu\Desktop\修士研究用\解析データ\{self.subject_num}\smarteye\gaze_direction"):
                os.makedirs(fr"C:\Users\tyasu\Desktop\修士研究用\解析データ\{self.subject_num}\smarteye\gaze_direction")
            plt.savefig(fr"C:\Users\tyasu\Desktop\修士研究用\解析データ\{self.subject_num}\smarteye\gaze_direction\multi_gaze_direction_{filename}.png")
        if show:
            plt.show()
        else:
            plt.close()
        
    def multi_analyze(self,titles:list=[],large_title_gaze_direction:str=None,store:bool=True,show:bool=True):
        n=len(self.smarteye_dfs)
        filename=""
        for experiment_num in self.experiment_nums:
            filename+=str(experiment_num)
            
        self.show_multi_gaze_direction(filename=filename,n=n,titles=titles,large_title=large_title_gaze_direction,store=store,show=show)