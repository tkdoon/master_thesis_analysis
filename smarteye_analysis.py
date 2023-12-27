import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import os
import math


class Smarteye_analysis():
    def __init__(self,subject_num:int,experiment_num:int=None,experiment_nums:list[int]=None,smarteye_df:pd.DataFrame=None,smarteye_dfs:list[pd.DataFrame]=None,path_to_data_dir=r"C:\Users\tyasu\Desktop\修士研究用") -> None:
        self.subject_num=subject_num
        self.experiment_num=experiment_num
        self.experiment_nums=experiment_nums
        self.smarteye_df=smarteye_df
        self.sampling_frequency=60#Hz
        if(smarteye_df is not None):
            self.smarteye_np=smarteye_df.values
            self.time_array=np.divide(self.smarteye_np[:,0].astype(int)-int(self.smarteye_np[0,0]),self.sampling_frequency)
        self.smarteye_dfs=smarteye_dfs
        self.path_to_data_dir=path_to_data_dir
        self.max_size=None
        self.mean_size=None
        self.min_size=None
        self.prc=None
        self.mean_distance=None
        self.analysis_path=os.path.join(path_to_data_dir,"解析データ",str(subject_num),"smarteye")
        
    def replace_zero_data(self,np_data:np.array,replace_method:str="linear")->np.array:
        """直前の値に置き換えるときはreplace_methodをpreviousに，線形補間するときは，replace_methodをlinearに"""
        if(replace_method=="previous"):
            # 直前の値に置き換える
            # データの値が0でないインデックスを取得
            nonzero_indices = np.where(np_data != 0)[0]

            # # データの値が0でない部分の平均値を計算
            # mean_value = np.mean(np_data[nonzero_indices])
            
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
        elif(replace_method=="linear"):
            # 0でない要素のインデックスを取得
            nonzero_indices = np.nonzero(np_data)[0]

            # 0でない要素が2つ以上存在する場合にのみ線形補間を行う
            if len(nonzero_indices) >= 2:
                # 0でない要素のインデックスを使用して、線形補間を行う
                np_data = np_data.astype(float)
                interpolated_values = np.interp(np.arange(len(np_data)), nonzero_indices, np_data[nonzero_indices])
                
                # 新しい配列に線形補間された値をセット
                interpolated = np.where(np_data == 0, interpolated_values, np_data)
                return interpolated
            else:
                raise ValueError("線形補間が行えません。非ゼロの要素が2つ以上必要です。")
        else:
            raise ValueError("replace_methodの値が違います．")
        
    
    def shorten_array(self,data:np.array,method:str):
        if(method == "non_zero_1d"):
            return data[np.nonzero(data)]
        elif(method == "q_non_zero"):
            return 
        elif(method == "non_zero_2d"):
            rows_to_delete=np.any(data == 0, axis=1)
            arr_without_zeros = data[~rows_to_delete]
            return arr_without_zeros

    def delete_inaccurate_data(self,np_data:np.array,q_data_index:int,threshold:float=0.3):
        # Q<thresholdのデータを消す
        prev_value = None
        
        for i in range(len(np_data)):
            if np_data[i, q_data_index] < threshold:
                if prev_value is not None:
                    np_data[i,:] = prev_value
                else:
                    # 0の前に値がない場合、そのままにしておくか、別の初期値を設定することもできます。
                    np_data[i,:] = 0  # 0の前に値がない場合、0のままにする例

            else:
                prev_value = np_data[i,:]

        return np_data
    
    def calculate_parameters(self):
        
        self.calculate_pupil_size()
        self.prc=self.calculate_PRC()
        self.calculate_eyelid_opening_distance()
        
        
    def calculate_pupil_size(self):    
        pupil_size_array=self.smarteye_np[:,523:525]
        fixed_data=self.delete_inaccurate_data(pupil_size_array,1,0.5)
        self.max_size=np.max(fixed_data[:,0])
        self.mean_size=np.mean(fixed_data[:,0])
        self.min_size=np.min(fixed_data[:,0])
        return {"maxsize":self.max_size, "mean_size":self.mean_size,"min_size":self.min_size,"data":fixed_data}
    
    def show_pupil_size(self,show:bool=True,store:bool=True,font_size:int=12):
        data=self.calculate_pupil_size()["data"]
        fig=plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        plt.plot(self.time_array,data[:,0],label="pupil size")
        plt.title('Pupil size')
        plt.xlabel('time (s)')
        plt.ylabel('pupil diameter(mm)')

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"pupil_size")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"pupil_size"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"pupil_size",f"pupil_size_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
            
    def calculate_eyelid_opening_distance(self):
        eyelid_data=self.smarteye_np[:,509:511]
        fixed_data=self.delete_inaccurate_data(eyelid_data,1,0.5)
        self.mean_distance=np.mean(fixed_data)
        return {"mean_distance": self.mean_distance,"data": fixed_data}
    
    def show_eyelid_opening_distance(self,show:bool=True,store:bool=True,font_size:int=12):
        data=self.calculate_eyelid_opening_distance()["data"]
        fig=plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        plt.plot(self.time_array,data[:,0],label="eyelid distance")
        plt.title('Eyelid distance')
        plt.xlabel('time (s)')
        plt.ylabel('eyelid distance(mm)')

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"eyelid")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"eyelid"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"eyelid",f"eyelid_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
    
    def show_gaze_direction(self,show:bool=True,store:bool=True,font_size:int=12):
        gaze_direction_array=self.smarteye_np[:,73:77]
        fig=plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        data=self.delete_inaccurate_data(gaze_direction_array,3,0.5)
        plt.plot(self.time_array,data[:,0],label="direction-x")
        plt.plot(self.time_array,data[:,1],label="direction-y")
        plt.plot(self.time_array,data[:,2],label="direction-z")
        plt.title('Gaze direction')
        plt.xlabel('time (s)')
        plt.ylabel('gaze direction')
        plt.legend()

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"gaze_direction")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"gaze_direction"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"gaze_direction",f"gaze_direction_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
            
    def calculate_PRC(self,gaze_angle=120,number_of_lattice=128):
        angle_each=gaze_angle/number_of_lattice
        angle_each_rad=math.radians(angle_each) 
        np_data=self.smarteye_np[:,109:111]

        ## どちらかが0のときその行を消す
        # nonzero_data=self.shorten_array(np_data,"non_zero_2d")
        # gaze_heading=nonzero_data[:,0]
        # gaze_pitch=nonzero_data[:,1]+math.pi/3
        # 0のとき直前の値に置き換える
        gaze_heading=self.replace_zero_data(np_data[:,0])
        gaze_pitch=self.replace_zero_data(np_data[:,1])+math.pi/3
        gaze_heading=np.where(gaze_heading>0,gaze_heading-2/3*math.pi,gaze_heading+4/3*math.pi)
        array_size=min(len(gaze_heading),len(gaze_pitch))
        def calculate_center_point():           
            count_array=np.zeros((number_of_lattice,number_of_lattice))
            for i in range(array_size):
                heading_index=int(gaze_heading[i]//angle_each_rad)
                pitch_index=int(gaze_pitch[i]//angle_each_rad)
                if(heading_index<0 or heading_index>=number_of_lattice or pitch_index<0 or pitch_index>=number_of_lattice):
                    pass
                else:
                    count_array[heading_index,pitch_index]+=1
                
            # 最大値を持つ要素のインデックスを取得
            max_index = np.argmax(count_array)
            # インデックスを2次元配列の形に変換
            mode_position = np.unravel_index(max_index, count_array.shape)
            center_point=(angle_each_rad*(mode_position[0]+1.0/2),angle_each_rad*(mode_position[1]+1.0/2))
            return center_point
        
        def calculate_angle_difference(center_point:tuple,vector:tuple):
            heading1_rad=center_point[0]
            pitch1_rad=center_point[1]
            heading2_rad=vector[0]
            pitch2_rad=vector[1]
            
            vector1 = (math.cos(pitch1_rad) * math.cos(heading1_rad),
               math.cos(pitch1_rad) * math.sin(heading1_rad),
               math.sin(pitch1_rad))
            vector2 = (math.cos(pitch2_rad) * math.cos(heading2_rad),
                    math.cos(pitch2_rad) * math.sin(heading2_rad),
                    math.sin(pitch2_rad))

            # 内積を計算
            dot_product = sum(a*b for a, b in zip(vector1, vector2))

            # 角度を計算
            angle_rad = math.acos(dot_product)
            return angle_rad
        
        count=0
        out_count=0
        center_point=calculate_center_point()
        for heading,pitch in zip(gaze_heading,gaze_pitch):
            angle_difference=calculate_angle_difference(center_point,(heading,pitch))
            if (angle_difference<=math.radians(8)):
                count+=1
            else:
                out_count+=1
        prc=count/array_size
        return prc
                
            
    def show_single_all(self,show=True,store=True,font_size:int=12):
        self.show_gaze_direction(show,store,font_size)
        self.show_pupil_size(show,store,font_size)
        self.show_eyelid_opening_distance(show,store,font_size)
            
    def show_multi_gaze_direction(self,filename,n,titles:list=[],large_title:str=None,store:bool=True,show:bool=True,font_size:int=12):
        def make1graph(time_array,gaze_direction_array,n,index,titles:list=[]):
            plt.subplot(n,1,index+1)
            data=self.delete_inaccurate_data(gaze_direction_array,3,0.3)
            
            plt.plot(time_array, data[:,0],label="direction-x")
            plt.plot(time_array, data[:,1],label="direction-y")
            plt.plot(time_array, data[:,2],label="direction-z")
            plt.xlabel('time(s)')
            plt.ylabel('gaze direction')
            plt.legend()
            plt.grid()

            if len(titles)!=0:
                plt.title(titles[index])
            
        fig = plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        for index,smarteye_log in enumerate(self.smarteye_dfs):
            smarteye_np=smarteye_log.values
            make1graph(np.divide(smarteye_np[:,0].astype(int)-int(smarteye_np[0,0]),self.sampling_frequency),smarteye_np[:,73:77],n,index,titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"gaze_direction")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"gaze_direction"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"gaze_direction",f"multi_gaze_direction_{filename}.png"))
        if show:
            plt.show()
        else:
            plt.close()
        
    def multi_analyze(self,titles:list=[],large_title_gaze_direction:str=None,store:bool=True,show:bool=True,font_size:int=12):
        n=len(self.smarteye_dfs)
        filename=""
        for experiment_num in self.experiment_nums:
            filename+=str(experiment_num)
            
        self.show_multi_gaze_direction(filename=filename,n=n,titles=titles,large_title=large_title_gaze_direction,store=store,show=show,font_size=font_size)
        
if __name__ == '__main__':
    import pandas as pd
    print(Smarteye_analysis(2,1,smarteye_df=pd.read_csv(r"C:\Users\tyasu\Desktop\修士研究用\実験データ\2\smarteye\1.log", sep='\t')).calculate_PRC())