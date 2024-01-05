import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import math


class DS_log_analysis():
    def __init__(self,subject_num,experiment_num:int=None,experiment_nums:list=None,df:pd.DataFrame=None,dfs:list=None,path_to_data_dir=r"C:\Users\tyasu\Desktop\修士研究用"):
        self.experiment_num = experiment_num
        self.subject_num =subject_num
        self.DS_log_df:pd.DataFrame=df
        if(df is not None):
            self.DS_log_np=self.DS_log_df.values
            self.times_np=self.DS_log_np[:,11]-self.DS_log_np[0,11]
            self.velocity_array=self.DS_log_np[:,34]
        self.analysis_path=os.path.join("解析データ",str(subject_num),"DS_log")
        if not os.path.exists(os.path.join(path_to_data_dir,"解析データ",str(subject_num))):
            os.makedirs(os.path.join(path_to_data_dir,"解析データ",str(subject_num)))
        if not os.path.exists(os.path.join(path_to_data_dir,self.analysis_path)):
            os.makedirs(os.path.join(path_to_data_dir,self.analysis_path))
        self.acceleration=None
        self.min_difference=None
        self.difference_variance=None
        self.difference_cv=None
        self.relative_velocity=None
        self.DS_log_dfs:list=dfs
        self.experiment_nums=experiment_nums
        self.path_to_data_dir=path_to_data_dir
        self.rms_acceleration=None
        self.acceleration_variance=None
        self.acceleration_cv=None
        self.rms_steering_angle=None
        self.accelerator_variance=None
        self.accelerator_mean=None
        self.accelerator_rms=None
        self.accelerator_vdv=None
        self.accelerator_cv=None
        self.brake_variance=None
        self.brake_mean=None
        self.brake_rms=None
        self.brake_vdv=None
        self.brake_cv=None
        self.max_velocity=None
        self.velocity_variance=None
        self.velocity_cv=None
        self.velocity_mean=None
        self.max_yaw=None
        self.yaw_variance=None
        self.yaw_cv=None
        
    def delete_velocity_zero(self,data):
        nonzero_indices = np.nonzero(self.velocity_array)[0]
        return data[nonzero_indices]
    
    def calculate_vdv(self,np_data):
        # 配列の4乗を計算
        array_pow_4 = np.power(np_data, 4)

        # 4乗平均を計算
        mean_pow_4 = np.mean(array_pow_4)

        # 4乗根を計算
        fourth_root = np.power(mean_pow_4, 1/4)
        return fourth_root
    
    def calculate_parameters(self,delete_zero:bool,initial_yaw):
        self.acceleration=np.sqrt(np.sum(self.DS_log_np[:,28:31]**2, axis=1))
        difference_array=self.calculate_distance_of_vehicles(car_position=self.DS_log_np[:,3:5]#(x,y)
                                                                   ,bus_position=self.DS_log_np[:,36:38]#(x,y)
                                                                   )
        if(delete_zero):
            self.rms_steering_angle=np.sqrt(np.mean(np.square(self.delete_velocity_zero(self.DS_log_np[:,1]))))
            dz_accelerator=self.delete_velocity_zero(self.DS_log_np[:,12])
            self.accelerator_variance=np.var(dz_accelerator)
            self.accelerator_mean=np.mean(dz_accelerator)
            self.accelerator_rms=np.sqrt(np.mean(np.square(dz_accelerator)))
            self.accelerator_vdv=self.calculate_vdv(dz_accelerator)
            self.accelerator_cv=np.std(dz_accelerator)/self.accelerator_mean
            dz_brake=self.delete_velocity_zero(self.DS_log_np[:,13])
            self.brake_variance=np.var(dz_brake)
            self.brake_mean=np.mean(dz_brake)
            self.brake_rms=np.sqrt(np.mean(np.square(dz_brake)))
            self.brake_vdv=self.calculate_vdv(dz_brake)
            self.brake_cv=np.std(dz_brake)/self.brake_mean
            dz_acceleration=self.delete_velocity_zero(self.acceleration)
            self.rms_acceleration = np.sqrt(np.mean(np.square(dz_acceleration)))
            self.acceleration_variance=np.var(dz_acceleration)
            self.acceleration_cv=np.std(dz_acceleration)/np.mean(dz_acceleration)
            dz_velocity=self.delete_velocity_zero(self.velocity_array)
            self.velocity_variance=np.var(dz_velocity)
            self.velocity_mean=np.mean(dz_velocity)
            self.velocity_cv=np.std(dz_velocity)/np.mean(dz_velocity)
            dz_yaw=self.delete_velocity_zero(self.DS_log_np[:,21])
            self.yaw_variance=np.var(dz_yaw)
            self.yaw_cv=np.std(dz_yaw)/np.mean(dz_yaw)
            dz_difference=self.delete_velocity_zero(difference_array)
            self.difference_variance=np.var(dz_difference)
            self.difference_cv=np.std(dz_difference)/np.mean(dz_difference)
        else:
            self.rms_steering_angle=np.sqrt(np.mean(np.square(self.DS_log_np[:,1])))
            self.accelerator_variance=np.var(self.DS_log_np[:,12])
            self.accelerator_mean=np.mean(self.DS_log_np[:,12])
            self.accelerator_rms=np.sqrt(np.mean(np.square(self.DS_log_np[:,12])))
            self.accelerator_vdv=self.calculate_vdv(self.DS_log_np[:,12])
            self.accelerator_cv=np.std(self.DS_log_np[:,12])/self.accelerator_mean
            self.brake_variance=np.var(self.DS_log_np[:,13])
            self.brake_mean=np.mean(self.DS_log_np[:,13])
            self.brake_rms=np.sqrt(np.mean(np.square(self.DS_log_np[:,13])))
            self.brake_vdv=self.calculate_vdv(self.DS_log_np[:,13])
            self.brake_cv=np.std(self.DS_log_np[:,13])/self.brake_mean
            self.rms_acceleration = np.sqrt(np.mean(np.square(self.acceleration)))
            self.acceleration_variance=np.var(self.acceleration)
            self.acceleration_cv=np.std(self.acceleration)/np.mean(self.acceleration)
            self.velocity_variance=np.var(self.velocity_array)
            self.velocity_mean=np.mean(self.velocity_array)
            self.velocity_cv=np.std(self.velocity_array)/np.mean(self.velocity_array)
            self.yaw_variance=np.var(self.DS_log_np[:,21])
            self.yaw_cv=np.std(self.DS_log_np[:,21])/np.mean(self.DS_log_np[:,21])
            self.difference_variance=np.var(difference_array)
            self.difference_cv=np.std(difference_array)/np.mean(difference_array)
        self.min_difference=np.min(difference_array)
        self.max_velocity=np.max(self.velocity_array)
        self.max_yaw=np.max(np.abs(self.DS_log_np[:,21]-initial_yaw))
        return

    def calculate_acceleration(self,velocity_array,times_np,acceleration_vector_array):
        # 速さの変化を計算
        # delta_speed = np.diff(velocity_array)  # 速さの差分を計算
        # delta_time = np.diff(times_np)  # 時間の差分を計算

        # # 加速度のnumpy配列を計算
        # acceleration = delta_speed / delta_time
        # # 最初の要素に対する加速度を計算
        # initial_acceleration = acceleration[0]
        # # 加速度の配列の最初に追加
        # self.acceleration = np.insert(acceleration, 0, initial_acceleration)
        self.acceleration=np.sqrt(np.sum(acceleration_vector_array**2, axis=1))
        return self.acceleration
        
    
    def show_acceleration(self,store:bool=True,show:bool=True,font_size:int=12):
        fig=plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        plt.plot(self.times_np,self.calculate_acceleration(self.velocity_array,self.times_np,self.DS_log_np[:,28:31]))
        plt.title('Absolute value of acceleration')
        plt.xlabel('time (s)')
        plt.ylabel('acceleration(m/s^2)')

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"acceleration")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"acceleration"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"acceleration",f"acceleration_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()     
            
    def show_multi_acceleration(self,n,filename,titles:list=[],large_title:str=None,store:bool=True,show:bool=True,font_size:int=12,sharey:bool=True):
        def make1graph(time_array,acceleration_array,n,index,titles:list=[]):
            plt.subplot(n,1,index+1)
            plt.plot(time_array, acceleration_array)
            plt.xlabel('time(s)')
            plt.ylabel('acceleration(m/s^2)')
            # plt.yticks([0,2,4,6])
            plt.grid()

            if len(titles)!=0:
                plt.title(titles[index])
            
        fig = plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        for index,DS_log in enumerate(self.DS_log_dfs):
            DS_log_np=DS_log.values
            make1graph(DS_log_np[:,11]-DS_log_np[0,11],self.calculate_acceleration(DS_log_np[:,34],DS_log_np[:,11]-DS_log_np[0,11],DS_log_np[:,28:31]),n,index,titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"acceleration")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"acceleration"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"acceleration",f"multi_acceleration_{filename}.png"))
        if show:
            plt.show()
        else:
            plt.close()
    
    def calculate_distance_of_vehicles(self,car_position,bus_position):
        difference_xy=bus_position-car_position
        difference=np.sqrt(np.sum(np.square(difference_xy),axis=1))
        return difference
    
    def calculate_relative_velocity(self,car_velocity,bus_velocity):
        relative_velocity=car_velocity-bus_velocity
        return relative_velocity
    
    def show_velocity(self,store:bool=True,show:bool=True,font_size:int=12):
        fig=plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        plt.plot(self.times_np,self.velocity_array)
        plt.title('Velocity')
        plt.xlabel('time (s)')
        plt.ylabel('velocity(m/s)')

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"velocity")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"velocity"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"velocity",f"velocity_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
    def show_multi_velocity(self,n,filename,titles:list=[],large_title:str=None,store:bool=True,show:bool=True,font_size:int=12):
        def make1graph(time_array,velocity_array,n,index,titles:list=[]):
            plt.subplot(n,1,index+1)
            plt.plot(time_array, velocity_array)
            plt.xlabel('time(s)')
            plt.ylabel('Velocity(m/s)')
            plt.grid()

            if len(titles)!=0:
                plt.title(titles[index])
            
        fig = plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        for index,DS_log in enumerate(self.DS_log_dfs):
            DS_log_np=DS_log.values
            make1graph(DS_log_np[:,11]-DS_log_np[0,11],DS_log_np[:,34],n,index,titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"velocity")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"velocity"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"velocity",f"multi_velocity_{filename}.png"))
        if show:
            plt.show()
        else:
            plt.close()
    
    def show_distance_of_vehicles(self,store:bool=True,show:bool=True,font_size:int=12):
        fig=plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        plt.plot(self.times_np,self.calculate_distance_of_vehicles(car_position=self.DS_log_np[:,3:5]#(x,y)
                                                                   ,bus_position=self.DS_log_np[:,36:38]#(x,y)
                                                                   ))
        plt.title('Distance between vehicles')
        plt.xlabel('time (s)')
        plt.ylabel('distance(m)')

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"distance_of_vehicles")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"distance_of_vehicles"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"distance_of_vehicles",f"distance_of_vehicles_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
    def show_multi_distance_of_vehicles(self,n,filename,titles:list=[],large_title:str=None,store:bool=True,show:bool=True,font_size:int=12):
        def make1graph(time_array,distance_of_vehicle_array,n,index,titles:list=[]):
            plt.subplot(n,1,index+1)
            plt.plot(time_array, distance_of_vehicle_array)
            plt.xlabel('time(s)')
            plt.ylabel('distance(m)')
            # plt.yticks([0,10,20,30,40,50])
            plt.grid()

            if len(titles)!=0:
                plt.title(titles[index])
            
        fig = plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        for index,DS_log in enumerate(self.DS_log_dfs):
            DS_log_np=DS_log.values
            make1graph(DS_log_np[:,11]-DS_log_np[0,11],self.calculate_distance_of_vehicles(DS_log_np[:,3:5],DS_log_np[:,36:38]),n,index,titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"distance_of_vehicles")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"distance_of_vehicles"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"distance_of_vehicles",f"multi_distance_of_vehicles_{filename}.png"))
        if show:
            plt.show()
        else:
            plt.close()
        
    def show_relative_velocity(self,store:bool=True,show:bool=True,font_size:int=12):
        fig=plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        plt.plot(self.times_np,self.calculate_relative_velocity(car_velocity=self.velocity_array,bus_velocity=self.DS_log_np[:,42]))
        plt.title('Relative velocity')
        plt.xlabel('time (s)')
        plt.ylabel('velocity(m/s)')

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"relative_velocity")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"relative_velocity"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"relative_velocity",f"relative_velocity_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
    def show_multi_relative_velocity(self,n,filename,titles:list=[],large_title:str=None,store:bool=True,show:bool=True,font_size:int=12):
        def make1graph(time_array,relative_velocity_array,n,index,titles:list=[]):
            plt.subplot(n,1,index+1)
            plt.plot(time_array, relative_velocity_array)
            plt.xlabel('time(s)')
            plt.ylabel('Velocity(m/s)')
            
            plt.grid()

            if len(titles)!=0:
                plt.title(titles[index])
            
        fig = plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        for index,DS_log in enumerate(self.DS_log_dfs):
            DS_log_np=DS_log.values
            make1graph(DS_log_np[:,11]-DS_log_np[0,11],self.calculate_relative_velocity(DS_log_np[:,34],DS_log_np[:,42]),n,index,titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"relative_velocity")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"relative_velocity"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"relative_velocity",f"multi_relative_velocity_{filename}.png"))
        if show:
            plt.show()
        else:
            plt.close()
        
    def show_accelerator_and_brake_pressure(self,store:bool=True,show:bool=True,font_size:int=12):
        fig=plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        plt.plot(self.times_np,self.low_pass_filter(self.DS_log_np[:,12],3,order=2),label="accelerator")
        plt.plot(self.times_np,self.DS_log_np[:,13],label="brake")
        plt.title('Accelerator and brake pressure')
        plt.xlabel('time (s)')
        plt.ylabel('pressure')
        plt.legend()  # 凡例を表示

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"accel_brake")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"accel_brake"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"accel_brake",f"accel_brake_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
            
    def show_multi_accelerator_and_brake_pressure(self,n,filename,titles:list=[],large_title:str=None,store:bool=True,show:bool=True,font_size:int=12):
        def make1graph(time_array,accelerator_array,brake_array,n,index,titles:list=[]):
            plt.subplot(n,1,index+1)
            plt.plot(time_array, accelerator_array,label="accelerator")
            plt.plot(time_array, brake_array,label="brake")
            plt.xlabel('time(s)')
            plt.ylabel('pressure')
            plt.grid()
            plt.legend()

            if len(titles)!=0:
                plt.title(titles[index])
            
        fig = plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        for index,DS_log in enumerate(self.DS_log_dfs):
            DS_log_np=DS_log.values
            make1graph(DS_log_np[:,11]-DS_log_np[0,11],self.low_pass_filter(DS_log_np[:,12],3,order=2),DS_log_np[:,13],n,index,titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"accel_brake")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"accel_brake"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"accel_brake",f"multi_accel_brake_{filename}.png"))      
        if show:
            plt.show()
        else:
            plt.close()
            
            
    def show_steering_angle(self,store:bool=True,show:bool=True,font_size:int=12):
        steering_angle_array=self.DS_log_np[:,1]
        fig=plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        plt.plot(self.times_np,steering_angle_array,label="steering angle")
        plt.title('Steering angle')
        plt.xlabel('time (s)')
        plt.ylabel('angle(deg)')

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"steering_angle")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"steering_angle"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"steering_angle",f"steering_angle_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
    def show_multi_steering_angle(self,n,filename,titles:list=[],large_title:str=None,store:bool=True,show:bool=True,font_size:int=12):
        def make1graph(time_array,steering_angle_array,n,index,titles:list=[]):
            plt.subplot(n,1,index+1)
            plt.plot(time_array, steering_angle_array,label="steering angle")
            plt.xlabel('time(s)')
            plt.ylabel('angle(deg)')
            # plt.yticks([-10,-5,0,5])
            plt.grid()

            if len(titles)!=0:
                plt.title(titles[index])
            
        fig = plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        for index,DS_log in enumerate(self.DS_log_dfs):
            DS_log_np=DS_log.values
            make1graph(DS_log_np[:,11]-DS_log_np[0,11],DS_log_np[:,1],n,index,titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"steering_angle")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"steering_angle"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"steering_angle",f"multi_steering_angle_{filename}.png"))      
        if show:
            plt.show()
        else:
            plt.close()
            
    def show_steering_torque_and_angle(self,store:bool=True,show:bool=True,font_size:int=12):
        x=self.times_np
        y1=self.DS_log_np[:,1]
        y2=self.DS_log_np[:,2]
        
        plt.rcParams["font.size"] = font_size
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
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"steering")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"steering"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"steering",f"steering_{self.experiment_num}.png"))  
        if show:
            plt.show()
        else:
            plt.close()
            
    def show_multi_steering_torque_and_angle(self,n,filename,titles:list=[],large_title:str=None,store:bool=True,show:bool=True,font_size:int=12):
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
        plt.rcParams["font.size"] = font_size
        fig, axes = plt.subplots(n, 1)
        for index,DS_log in enumerate(self.DS_log_dfs):
            DS_log_np=DS_log.values
            make1graph(DS_log_np[:,11]-DS_log_np[0,11],DS_log_np[:,1],DS_log_np[:,2],axes[index],titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"steering")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"steering"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"steering",f"multi_steering_{filename}.png"))   
        if show:
            plt.show()
        else:
            plt.close()
            
    def show_yaw(self,store:bool=True,show:bool=True,font_size:int=12):
        fig=plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        plt.plot(self.times_np,self.DS_log_np[:,21]-self.DS_log_np[0,21],label="yaw angle")
        plt.title('Yaw angle')
        plt.xlabel('time (s)')
        plt.ylabel('angle(deg)')

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"yaw_angle")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"yaw_angle"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"yaw_angle",f"yaw_angle_{self.experiment_num}.png"))
        if show:
            plt.show()
        else:
            plt.close()
            
            
    def show_multi_yaw(self,n,filename,titles:list=[],large_title:str=None,store:bool=True,show:bool=True,font_size:int=12):
        def make1graph(time_array,yaw_array,n,index,titles:list=[]):
            plt.subplot(n,1,index+1)
            plt.plot(time_array, yaw_array,label="yaw angle")
            plt.xlabel('time(s)')
            plt.ylabel('angle(deg)')
            # plt.yticks([-10,-5,0,5])
            plt.grid()

            if len(titles)!=0:
                plt.title(titles[index])
            
        fig = plt.figure(figsize=(16,12))
        plt.rcParams["font.size"] = font_size
        for index,DS_log in enumerate(self.DS_log_dfs):
            DS_log_np=DS_log.values
            make1graph(DS_log_np[:,11]-DS_log_np[0,11],DS_log_np[:,21],n,index,titles)
        if large_title:
            plt.suptitle(large_title)
        plt.tight_layout()

        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,self.analysis_path,"yaw_angle")):
                os.makedirs(os.path.join(self.path_to_data_dir,self.analysis_path,"yaw_angle"))
            plt.savefig(os.path.join(self.path_to_data_dir,self.analysis_path,"yaw_angle",f"multi_yaw_angle_{filename}.png"))      
        if show:
            plt.show()
        else:
            plt.close()
        
    def show_all(self,store:bool=True,show:bool=True,font_size:int=12):
        self.show_velocity(store,show,font_size)
        self.show_distance_of_vehicles(store,show,font_size)
        self.show_relative_velocity(store,show,font_size)
        self.show_accelerator_and_brake_pressure(store,show,font_size)
        self.show_steering_torque_and_angle(store,show,font_size)
        self.show_acceleration(store,show,font_size)
        self.show_steering_angle(store,show,font_size)
        self.show_yaw(store,show,font_size)
        
    def multi_analyze_all(self,titles:list=[],large_title_velocity:str=None,large_title_distance_of_vehicles:str=None,large_title_relative_velocity:str=None,large_title_accelerator_and_brake:str=None,large_title_steering_torque_and_angle:str=None,large_title_acceleration:str=None,large_title_steering_angle:str=None,large_title_yaw:str=None,store:bool=True,show:bool=True,font_size:int=12):
        n=len(self.DS_log_dfs)
        filename=""
        for experiment_num in self.experiment_nums:
            filename+=str(experiment_num)
        self.show_multi_velocity(n=n,filename=filename,titles=titles,large_title=large_title_velocity,store=store,show=show,font_size=font_size)
        self.show_multi_distance_of_vehicles(n=n,filename=filename,titles=titles,large_title=large_title_distance_of_vehicles,store=store,show=show,font_size=font_size)
        self.show_multi_relative_velocity(n=n,filename=filename,titles=titles,large_title=large_title_relative_velocity,store=store,show=show,font_size=font_size)
        self.show_multi_accelerator_and_brake_pressure(n=n,filename=filename,titles=titles,large_title=large_title_accelerator_and_brake,store=store,show=show,font_size=font_size)
        self.show_multi_steering_torque_and_angle(n=n,filename=filename,titles=titles,large_title=large_title_steering_torque_and_angle,store=store,show=show,font_size=font_size)
        self.show_multi_acceleration(n=n,filename=filename,titles=titles,large_title=large_title_acceleration,store=store,show=show,font_size=font_size)
        self.show_multi_steering_angle(n=n,filename=filename,titles=titles,large_title=large_title_steering_angle,store=store,show=show,font_size=font_size)
        self.show_multi_yaw(n=n,filename=filename,titles=titles,large_title=large_title_yaw,store=store,show=show,font_size=font_size)
        
    def low_pass_filter(self,data:np.array,cutoff_frequency:int,sampling_rate:int=120,order:int=2):
        from scipy.signal import butter, lfilter,filtfilt
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_frequency / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
                
    
        
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