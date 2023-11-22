import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib


class DS_log_analysis():
    def __init__(self,subject_num,experiment_num:int=None,experiment_nums:list=None,df:pd.DataFrame=None,dfs:list=None,path_to_data_dir=r"C:\Users\tyasu\Desktop\修士研究用"):
        self.experiment_num = experiment_num
        self.subject_num =subject_num
        self.DS_log_df:pd.DataFrame=df
        if(df is not None):
            self.DS_log_np=self.DS_log_df.values
            self.times_np=self.DS_log_np[:,11]-self.DS_log_np[0,11]
            self.velocity_array=self.DS_log_np[:,34]
        if not os.path.exists(os.path.join(self.path_to_data_dir,os.path.join(self.path_to_data_dir,fr"解析データ\{subject_num}"))):
            os.makedirs(os.path.join(self.path_to_data_dir,os.path.join(self.path_to_data_dir,fr"解析データ\{subject_num}")))
        if not os.path.exists(os.path.join(self.path_to_data_dir,os.path.join(self.path_to_data_dir,fr"解析データ\{subject_num}\DS_log"))):
            os.makedirs(os.path.join(self.path_to_data_dir,os.path.join(self.path_to_data_dir,fr"解析データ\{subject_num}\DS_log")))
        self.acceleration=None
        self.difference=None
        self.relative_velocity=None
        self.DS_log_dfs:list=dfs
        self.experiment_nums=experiment_nums
        self.self.path_to_data_dir=self.path_to_data_dir
            
            
    def calculate_acceleration(self):
        # 速さの変化を計算
        delta_speed = np.diff(self.velocity_array)  # 速さの差分を計算
        delta_time = np.diff(self.times_np)  # 時間の差分を計算

        # 加速度のnumpy配列を計算
        acceleration = delta_speed / delta_time
        # 最初の要素に対する加速度を計算
        initial_acceleration = acceleration[0]

        # 加速度の配列の最初に追加
        self.acceleration = np.insert(acceleration, 0, initial_acceleration)
        
        return self.acceleration
    
    def calculate_distance_of_vehicles(self,car_position,bus_position):
        difference_xy=bus_position-car_position
        difference=np.sqrt(np.sum(np.square(difference_xy),axis=1))
        return difference
    
    def calculate_relative_velocity(self,car_velocity,bus_velocity):
        relative_velocity=car_velocity-bus_velocity
        return relative_velocity
    
    def show_velocity(self,store:bool=True,show:bool=True):
        fig=plt.figure(figsize=(16,12))
        plt.plot(self.times_np,self.velocity_array)
        plt.title('Velocity')
        plt.xlabel('time (s)')
        plt.ylabel('velocity(m/s)')
        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\velocity")):
                os.makedirs(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\velocity"))
            plt.savefig(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\velocity\velocity_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
    def show_multi_velocity(self,n,filename,titles:list=[],large_title:str=None,store:bool=True,show:bool=True):
        def make1graph(time_array,velocity_array,n,index,titles:list=[]):
            plt.subplot(n,1,index+1)
            plt.plot(time_array, velocity_array)
            plt.xlabel('time(s)')
            plt.ylabel('Velocity(m/s)')
            plt.grid()

            if len(titles)!=0:
                plt.title(titles[index])
            
        fig = plt.figure(figsize=(16,12))
        for index,DS_log in enumerate(self.DS_log_dfs):
            DS_log_np=DS_log.values
            make1graph(DS_log_np[:,11]-DS_log_np[0,11],DS_log_np[:,34],n,index,titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()
        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\velocity")):
                os.makedirs(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\velocity"))
            plt.savefig(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\velocity\multi_velocity_{filename}.png"))
        if show:
            plt.show()
        else:
            plt.close()
    
    def show_distance_of_vehicles(self,store:bool=True,show:bool=True):
        fig=plt.figure(figsize=(16,12))
        plt.plot(self.times_np,self.calculate_distance_of_vehicles(car_position=self.DS_log_np[:,3:5]#(x,y)
                                                                   ,bus_position=self.DS_log_np[:,36:38]#(x,y)
                                                                   ))
        plt.title('Distance between vehicles')
        plt.xlabel('time (s)')
        plt.ylabel('distance(m)')
        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\distance_of_vehicles")):
                os.makedirs(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\distance_of_vehicles"))
            plt.savefig(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\distance_of_vehicles\distance_of_vehicles_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
    def show_multi_distance_of_vehicles(self,n,filename,titles:list=[],large_title:str=None,store:bool=True,show:bool=True):
        def make1graph(time_array,distance_of_vehicle_array,n,index,titles:list=[]):
            plt.subplot(n,1,index+1)
            plt.plot(time_array, distance_of_vehicle_array)
            plt.xlabel('time(s)')
            plt.ylabel('Distance of Vehicles(m)')
            plt.grid()

            if len(titles)!=0:
                plt.title(titles[index])
            
        fig = plt.figure(figsize=(16,12))
        for index,DS_log in enumerate(self.DS_log_dfs):
            DS_log_np=DS_log.values
            make1graph(DS_log_np[:,11]-DS_log_np[0,11],self.calculate_distance_of_vehicles(DS_log_np[:,3:5],DS_log_np[:,36:38]),n,index,titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()
        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\distance_of_vehicles")):
                os.makedirs(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\distance_of_vehicles"))
            plt.savefig(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\distance_of_vehicles\multi_distance_of_vehicles_{filename}.png"))
        if show:
            plt.show()
        else:
            plt.close()
        
    def show_relative_velocity(self,store:bool=True,show:bool=True):
        fig=plt.figure(figsize=(16,12))
        plt.plot(self.times_np,self.calculate_relative_velocity(car_velocity=self.velocity_array,bus_velocity=self.DS_log_np[:,42]))
        plt.title('Relative velocity')
        plt.xlabel('time (s)')
        plt.ylabel('velocity(m/s)')
        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\relative_velocity")):
                os.makedirs(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\relative_velocity"))
            plt.savefig(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\relative_velocity\relative_velocity_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
    def show_multi_relative_velocity(self,n,filename,titles:list=[],large_title:str=None,store:bool=True,show:bool=True):
        def make1graph(time_array,relative_velocity_array,n,index,titles:list=[]):
            plt.subplot(n,1,index+1)
            plt.plot(time_array, relative_velocity_array)
            plt.xlabel('time(s)')
            plt.ylabel('Relative Velocity(m/s)')
            plt.grid()

            if len(titles)!=0:
                plt.title(titles[index])
            
        fig = plt.figure(figsize=(16,12))
        for index,DS_log in enumerate(self.DS_log_dfs):
            DS_log_np=DS_log.values
            make1graph(DS_log_np[:,11]-DS_log_np[0,11],self.calculate_relative_velocity(DS_log_np[:,34],DS_log_np[:,42]),n,index,titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()
        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\relative_velocity")):
                os.makedirs(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\relative_velocity"))
            plt.savefig(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\relative_velocity\multi_relative_velocity_{filename}.png"))
        if show:
            plt.show()
        else:
            plt.close()
        
    def show_accelerator_and_brake_pressure(self,store:bool=True,show:bool=True):
        fig=plt.figure(figsize=(16,12))
        plt.plot(self.times_np,self.DS_log_np[:,12],label="accelerator")
        plt.plot(self.times_np,self.DS_log_np[:,13],label="brake")
        plt.title('Accelerator and brake pressure')
        plt.xlabel('time (s)')
        plt.ylabel('pressure')
        plt.legend()  # 凡例を表示
        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\accel_brake")):
                os.makedirs(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\accel_brake"))
            plt.savefig(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\accel_brake\accel_brake_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
            
    def show_multi_accelerator_and_brake_pressure(self,n,filename,titles:list=[],large_title:str=None,store:bool=True,show:bool=True):
        def make1graph(time_array,accelerator_array,brake_array,n,index,titles:list=[]):
            plt.subplot(n,1,index+1)
            plt.plot(time_array, accelerator_array,label="accelerator")
            plt.plot(time_array, brake_array,label="brake")
            plt.xlabel('time(s)')
            plt.ylabel('accelerator and brake pressure(m/s)')
            plt.grid()
            plt.legend()

            if len(titles)!=0:
                plt.title(titles[index])
            
        fig = plt.figure(figsize=(16,12))
        for index,DS_log in enumerate(self.DS_log_dfs):
            DS_log_np=DS_log.values
            make1graph(DS_log_np[:,11]-DS_log_np[0,11],DS_log_np[:,12],DS_log_np[:,13],n,index,titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()
        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\accel_brake")):
                os.makedirs(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\accel_brake"))
            plt.savefig(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\accel_brake\multi_accel_brake_{filename}.png"))      
        if show:
            plt.show()
        else:
            plt.close()
            
            
    def show_steering_torque_and_angle(self,store:bool=True,show:bool=True):
        x=self.times_np
        y1=self.DS_log_np[:,1]
        y2=self.DS_log_np[:,2]
        
        fig, ax1 = plt.subplots()
        fig.set_size_inches(16, 12, forward=True)
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('deg', color='tab:blue')
        ax1.plot(x, y1, color='tab:blue', label='steering angle(deg)')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Nm', color='tab:red')
        ax2.plot(x, y2, color='tab:red', label='steering torque(Nm)')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        plt.legend()
        # fig.tight_layout()
        plt.title('Steering angle and torque')
        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\steering")):
                os.makedirs(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\steering"))
            plt.savefig(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\steering\steering_{self.experiment_num}.png"))  
        if show:
            plt.show()
        else:
            plt.close()
            
    def show_multi_steering_torque_and_angle(self,n,filename,titles:list=[],large_title:str=None,store:bool=True,show:bool=True):
        def make1graph(time_array,accelerator_array,brake_array,ax,titles:list=[]):
            x=time_array
            y1=accelerator_array
            y2=brake_array
            
            ax.set_xlabel('time (s)')
            ax.set_ylabel('deg', color='tab:blue')
            ax.plot(x, y1, color='tab:blue', label='steering angle(deg)')
            ax.tick_params(axis='y', labelcolor='tab:blue')

            ax2 = ax.twinx()
            ax2.set_ylabel('Nm', color='tab:red')
            ax2.plot(x, y2, color='tab:red', label='steering torque(Nm)')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            plt.legend()
            

            if len(titles)!=0:
                plt.title(titles[index])
            
        fig = plt.figure(figsize=(16,12))
        fig, axes = plt.subplots(n, 1)
        for index,DS_log in enumerate(self.DS_log_dfs):
            DS_log_np=DS_log.values
            make1graph(DS_log_np[:,11]-DS_log_np[0,11],DS_log_np[:,1],DS_log_np[:,2],axes[index],titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()
        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\steering")):
                os.makedirs(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\steering"))
            plt.savefig(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\DS_log\steering\multi_steering_{filename}.png"))   
        if show:
            plt.show()
        else:
            plt.close()
        
    def analyze_all(self,store:bool=True,show:bool=True):
        self.show_velocity(store,show)
        self.show_distance_of_vehicles(store,show)
        self.show_relative_velocity(store,show)
        self.show_accelerator_and_brake_pressure(store,show)
        self.show_steering_torque_and_angle(store,show)
        
    def multi_analyze_all(self,titles:list=[],large_title_velocity:str=None,large_title_distance_of_vehicles:str=None,large_title_relative_velocity:str=None,large_title_accelerator_and_brake:str=None,large_title_steering_torque_and_angle:str=None,store:bool=True,show:bool=True):
        n=len(self.DS_log_dfs)
        filename=""
        for experiment_num in self.experiment_nums:
            filename+=str(experiment_num)
        self.show_multi_velocity(n=n,filename=filename,titles=titles,large_title=large_title_velocity,store=store,show=show)
        self.show_multi_distance_of_vehicles(n=n,filename=filename,titles=titles,large_title=large_title_distance_of_vehicles,store=store,show=show)
        self.show_multi_relative_velocity(n=n,filename=filename,titles=titles,large_title=large_title_relative_velocity,store=store,show=show)
        self.show_multi_accelerator_and_brake_pressure(n=n,filename=filename,titles=titles,large_title=large_title_accelerator_and_brake,store=store,show=show)
        self.show_multi_steering_torque_and_angle(n=n,filename=filename,titles=titles,large_title=large_title_steering_torque_and_angle,store=store,show=show)
        
    def low_pass_filter(self,data:np.array,cutoff_frequency:int,sampling_rate:int=120,order:int=5):
        from scipy.signal import butter, lfilter
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_frequency / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = lfilter(b, a, data)
        return filtered_data
                
    
        
    def calculate_variance(self):#分散
        return
        
        
if __name__=="__main__":
    data=pd.read_csv(r"実験データ\1\ds_log\ds1.csv", header=6,encoding="shift-jis").values[:,12]
    filtered_data=DS_log_analysis(1,1).low_pass_filter(data,3,order=10)
    t = pd.read_csv(r"実験データ\1\ds_log\ds1.csv", header=6,encoding="shift-jis").values[:,11]
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(t, data, 'b-', label='Original Data')
    plt.plot(t, filtered_data, 'r-', label='Filtered Data')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()