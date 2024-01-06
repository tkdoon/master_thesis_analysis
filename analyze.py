from time_fitting import Time_fitting
from ECG_analysis import ECG_analysis
from EMG_analysis import emg_analysis,check_burst,EMG_analysis
from DS_log_analysis import DS_log_analysis
from smarteye_analysis import Smarteye_analysis
from nasatlx_analysis import Nasatlx_analysis
from info import categorized_experiment_num,create_experiment_info
import json
import os
import pandas as pd

def main(subject_num):
    print("被験者番号:",subject_num)
    delay=48 if subject_num==2 else 35 if subject_num==1 else 62 if subject_num==3 else 52
    path_to_data_dir=r"C:\Users\tyasu\Desktop\修士研究用"
    # path_to_data_dir=r"/content/drive/MyDrive"

    dfs_list=[]
    parameter_list=[]
    emg_obj=EMG_analysis(subject_num=subject_num)
    for experiment_num in range(1,24):
        experiment_num=experiment_num+20
        if experiment_num==4:
            break
        print("実験番号:",experiment_num)
        dfs=read_df(path_to_data_dir=path_to_data_dir,subject_num=subject_num,experiment_num=experiment_num)
        experiment_info=create_experiment_info(dfs["DS_log_df"],experiment_num)
        print(experiment_info)
        time_obj=Time_fitting(experiment_num=experiment_num,subject_num=subject_num,before_time=experiment_info["before_time"],after_time=experiment_info["after_time"],delay=delay,path_to_data_dir=path_to_data_dir,**dfs)
        polymate_df,before10s_df=time_obj.time_fitting_polymate(check_delay=False)
        smarteye_df=time_obj.time_fitting_smarteye(delete_during_stop=False)
        smarteye_df_without_stoptime=time_obj.time_fitting_smarteye(delete_during_stop=True)
        DS_log_df=time_obj.time_fitting_DS_log()
        # dfs_list.append([polymate_df, smarteye_df, DS_log_df])
        # polymate_dfs4emg=[]
        # for emg_time in experiment_info["emg"]:
        #     polymate_dfs4emg.append(time_obj.time_fitting_polymate4emg(emg_time["before_time"],emg_time["after_time"]))
        # emg_value_list=[]
        # for idx,polymate_df4emg in enumerate(polymate_dfs4emg):
        #     emg_percent=emg_obj.main(polymate_df4emg,"vdv","mean")
        #     emg_value_list.append(emg_percent)
        #     emg_obj.show_emg_graph(emg_df=polymate_df,experiment_num=experiment_num,path_to_data_dir=path_to_data_dir,store=True,show=False,n=idx)
        #     emg_obj.show_polymate_graph(polymate_df,experiment_num=experiment_num,path_to_data_dir=path_to_data_dir,store=True,show=False,n=idx)
        # time_obj.cut_video(output_file_name = f'cut{experiment_num}.mp4')
        ecg_obj=ECG_analysis(subject_num=subject_num,experiment_num=experiment_num,ecg_df=polymate_df,path_to_data_dir=path_to_data_dir,before10s_df=before10s_df)
        res,before10s_res=ecg_obj.analyze_single(False,True,fix=True,font_size=16)

        # emg_res=emg_analysis(polymate_df,experiment_num=experiment_num,subject_num=subject_num,show=False,path_to_data_dir=path_to_data_dir)
        # show_emg_graph(polymate_df,subject_num=subject_num,experiment_num=experiment_num,show=False)
        # burst_list=check_burst(polymate_df,subject_num=subject_num,experiment_num=experiment_num,path_to_data_dir=path_to_data_dir)
        # burst_times=len(burst_list)
        # burst_seconds_mean=None
        # burst_seconds_sum=None
        # if(burst_times !=0):
        #     seconds_list=[]
        #     for burst in burst_list:
        #         seconds_list.append(burst[1]-burst[0])
        #     burst_seconds_sum=sum(seconds_list)
        #     burst_seconds_mean=burst_seconds_sum / burst_times

        ds_log_obj=DS_log_analysis(experiment_num=experiment_num,subject_num=subject_num,df=DS_log_df,path_to_data_dir=path_to_data_dir)
        ds_log_obj.show_all(show=False)
        ds_log_obj.calculate_parameters(experiment_info["delete_zero"],dfs["DS_log_df"].values[0,21])
        # smarteye_obj=Smarteye_analysis(subject_num=subject_num,experiment_num=experiment_num,smarteye_df=smarteye_df,path_to_data_dir=path_to_data_dir)
        # smarteye_obj.show_single_all(show=False)
        # smarteye_obj.calculate_parameters()
        # smarteye_obj_without_stoptime=Smarteye_analysis(subject_num=subject_num,experiment_num=experiment_num,smarteye_df=smarteye_df_without_stoptime,path_to_data_dir=path_to_data_dir)
        # smarteye_obj_without_stoptime.calculate_parameters()
        
        nasatlx_obj=Nasatlx_analysis(subject_num=subject_num,path_to_data_dir=path_to_data_dir)
        dictionary=nasatlx_obj.open_file()

        result=json.loads(dictionary[f"experiment{experiment_num}"])
        frustration=int(result["calculatedResult"]["results_rating"][5])
        overall_workload=float(result["calculatedResult"]["results_overall"])
        
        
        # parameter_list.append([experiment_num,ecg_obj.hf,ecg_obj.lf,ecg_obj.hf_lf_rate,ecg_obj.normalized_max_rri,ecg_obj.normalized_min_rri,ecg_obj.normalized_mean_rri,ecg_obj.normalized_sdnn,ecg_obj.used_df_num,ecg_obj.outlier_num,ecg_obj.outlier_rate,ds_log_obj.accelerator_mean,ds_log_obj.accelerator_variance,ds_log_obj.accelerator_rms,ds_log_obj.accelerator_vdv,ds_log_obj.accelerator_cv,ds_log_obj.brake_mean,ds_log_obj.brake_variance,ds_log_obj.brake_rms,ds_log_obj.brake_vdv,ds_log_obj.brake_cv,ds_log_obj.min_difference,ds_log_obj.difference_variance,ds_log_obj.difference_cv,ds_log_obj.rms_acceleration,ds_log_obj.acceleration_variance,ds_log_obj.acceleration_cv,ds_log_obj.rms_steering_angle,ds_log_obj.max_velocity,ds_log_obj.velocity_variance,ds_log_obj.velocity_mean,ds_log_obj.velocity_cv,ds_log_obj.max_yaw,ds_log_obj.yaw_variance,ds_log_obj.yaw_cv,smarteye_obj_without_stoptime.mean_distance,smarteye_obj_without_stoptime.max_size,smarteye_obj_without_stoptime.min_size,smarteye_obj_without_stoptime.mean_size,smarteye_obj_without_stoptime.prc,frustration,overall_workload,*emg_value_list])
        
        for i,time in enumerate(experiment_info["short_time"]):
            time_obj_short=Time_fitting(experiment_num=experiment_num,subject_num=subject_num,before_time=time["before_time"],after_time=time["after_time"],delay=delay,path_to_data_dir=path_to_data_dir,**dfs)
            short_polymate_df,_=time_obj_short.time_fitting_polymate(check_delay=False)
            short_smarteye_df_without_stoptime=time_obj_short.time_fitting_smarteye(delete_during_stop=True)
            short_DS_log_df=time_obj_short.time_fitting_DS_log()
            # time_obj_short.cut_video(output_file_name = f'cut{experiment_num}short{i}.mp4')
            short_ecg_obj=ECG_analysis(subject_num=subject_num,experiment_num=experiment_num,ecg_df=short_polymate_df,path_to_data_dir=path_to_data_dir,before10s_df=before10s_df)
            short_ecg_obj.get_result(before10s_res=before10s_res,fix=True)
            short_ds_log_obj=DS_log_analysis(experiment_num=experiment_num,subject_num=subject_num,df=short_DS_log_df,path_to_data_dir=path_to_data_dir)
            short_ds_log_obj.calculate_parameters(experiment_info["delete_zero"],dfs["DS_log_df"].values[0,21])
            # short_smarteye_obj=Smarteye_analysis(subject_num=subject_num,experiment_num=experiment_num,smarteye_df=short_smarteye_df_without_stoptime,path_to_data_dir=path_to_data_dir)
            # short_smarteye_obj.calculate_parameters()
            # parameter_list.append([experiment_num,short_ecg_obj.hf,short_ecg_obj.lf,short_ecg_obj.hf_lf_rate,short_ecg_obj.normalized_max_rri,short_ecg_obj.normalized_min_rri,short_ecg_obj.normalized_mean_rri,short_ecg_obj.normalized_sdnn,short_ecg_obj.used_df_num,short_ecg_obj.outlier_num,short_ecg_obj.outlier_rate,short_ds_log_obj.accelerator_mean,short_ds_log_obj.accelerator_variance,short_ds_log_obj.accelerator_rms,short_ds_log_obj.accelerator_vdv,short_ds_log_obj.accelerator_cv,short_ds_log_obj.brake_mean,short_ds_log_obj.brake_variance,short_ds_log_obj.brake_rms,short_ds_log_obj.brake_vdv,short_ds_log_obj.brake_cv,short_ds_log_obj.min_difference,short_ds_log_obj.difference_variance,short_ds_log_obj.difference_cv,short_ds_log_obj.rms_acceleration,short_ds_log_obj.acceleration_variance,short_ds_log_obj.acceleration_cv,short_ds_log_obj.rms_steering_angle,short_ds_log_obj.max_velocity,short_ds_log_obj.velocity_variance,short_ds_log_obj.velocity_mean,short_ds_log_obj.velocity_cv,short_ds_log_obj.max_yaw,short_ds_log_obj.yaw_variance,short_ds_log_obj.yaw_cv,short_smarteye_obj.mean_distance,short_smarteye_obj.max_size,short_smarteye_obj.min_size,short_smarteye_obj.mean_size,short_smarteye_obj.prc,frustration,overall_workload,emg_value_list[i]])
            
    
    large_key_list=[None,"ECG",None,None,None,None,None,None,None,None,None,"DS log",None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,"smarteye",None,None,None,None,"NASA TLX",None,"EMG"]
    key_list=["実験番号","hf","lf","hf/lf","正規化最大RRI","正規化最小RRI","正規化平均RRI","正規化rriの標準偏差","使用した心電図の番号(0~4,0=電極間,1,2=手首電極ref間,3,4=咬筋電極ref間)","外れ値の数","外れ値の割合","アクセルの踏み込み量平均","アクセルの踏み込み量分散","アクセルの踏み込み量rms","アクセルの踏み込み量vdv","アクセルの踏み込み量変動係数","ブレーキの踏み込み量平均","ブレーキの踏み込み量分散","ブレーキの踏み込み量rms","ブレーキの踏み込み量vdv","ブレーキの踏み込み量変動係数","最小車間距離","車間距離の分散","車間距離の変動係数","加速度の二乗平均","加速度の分散","加速度の変動係数","ステアリング角の二乗平均","最大速度","速度の分散","平均速度","速度の変動係数","ヨー角の最大値","ヨー角の分散","ヨー角の変動係数","瞼の平均距離","瞳孔の最大直径","瞳孔の最小直径","瞳孔の平均直径","PRC","フラストレーション","全体のワークロード","4乗平均4乗根/平均"]
    write_parameters_to_csv(os.path.join(path_to_data_dir,fr"解析データ/{subject_num}/parameters.csv"),large_key_list,key_list,*parameter_list)
    
    for experiment_nums in categorized_experiment_num:
        polymate_dfs=[]
        smarteye_dfs=[]
        DS_log_dfs=[]
        filename=""
        for experiment_num in experiment_nums["num"]:
            polymate_dfs.append(dfs_list[experiment_num-1][0])
            smarteye_dfs.append(dfs_list[experiment_num-1][1])
            DS_log_dfs.append(dfs_list[experiment_num-1][2])
            filename+=str(experiment_num)
        ECG_analysis(subject_num=subject_num,experiment_nums=experiment_nums["num"],ecg_dfs=polymate_dfs,path_to_data_dir=path_to_data_dir).show_multiple(titles=experiment_nums["title"],large_title_rri=experiment_nums["large_title_rri"],large_title_fft=experiment_nums["large_title_fft"],large_title_mean_beat=experiment_nums["large_title_mean_beat"],show=False,store=True,fix=True,font_size=16)
        DS_log_analysis(subject_num=subject_num,experiment_nums=experiment_nums["num"],dfs=DS_log_dfs,path_to_data_dir=path_to_data_dir).multi_analyze_all(titles=experiment_nums["title"],large_title_distance_of_vehicles=experiment_nums["large_title_distance_of_vehicles"],large_title_relative_velocity=experiment_nums["large_title_relative_velocity"],large_title_accelerator_and_brake=experiment_nums["large_title_accelerator_and_brake"],large_title_steering_torque_and_angle=experiment_nums["large_title_steering_torque_and_angle"],large_title_acceleration=experiment_nums["large_title_acceleration"],large_title_velocity=experiment_nums["large_title_velocity"],large_title_steering_angle=experiment_nums["large_title_steering_angle"],store=True,show=False,font_size=16)
        Smarteye_analysis(subject_num=subject_num,experiment_nums=experiment_nums["num"],smarteye_dfs=smarteye_dfs,path_to_data_dir=path_to_data_dir).multi_analyze(titles=experiment_nums["title"],large_title_gaze_direction=experiment_nums["large_title_gaze_direction"],store=True,show=False)
        Nasatlx_analysis(subject_num=subject_num,path_to_data_dir=path_to_data_dir).show_workload_frustration(experiment_nums=experiment_nums["num"],titles=experiment_nums["title"],large_title=experiment_nums["large_title_nasatlx"],show=False,store=True)
        emg_obj.show_multi_emg_graph(experiment_nums=experiment_nums["num"],filename=filename,titles=experiment_nums["title"],large_title=experiment_nums["large_title_emg"],show=False,store=True)
        
def write_parameters_to_csv(csv_path,*args):
    """可変長引数の中身はkey,value1,value2,...の形で渡してください"""
    import csv
    with open(csv_path,'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # データをCSVファイルに書き込む
        csv_writer.writerows(args)

def read_df(path_to_data_dir,subject_num,experiment_num):
        path_to_polymate_log = os.path.join(path_to_data_dir,"実験データ",f"{subject_num}","polymate","new",f"実験{experiment_num}.CSV")
        path_to_smarteye_log=os.path.join(path_to_data_dir,"実験データ",f"{subject_num}","smarteye",f"{experiment_num}.log")
        path_to_DS_log=os.path.join(path_to_data_dir,"実験データ",f"{subject_num}","ds_log",f"ds{experiment_num}.csv")
        smarteye_df:pd.DataFrame=pd.read_csv(path_to_smarteye_log, sep='\t')  # 列名はタブ文字で区切られている
        polymate_df:pd.DataFrame=pd.read_csv(path_to_polymate_log, header=3)
        DS_log_df:pd.DataFrame=pd.read_csv(path_to_DS_log, header=5,encoding="shift-jis")
        
        return {"smarteye_df":smarteye_df,"polymate_df":polymate_df,"DS_log_df":DS_log_df}

if __name__ == '__main__':
    main(4)