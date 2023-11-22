from time_fitting import Time_fitting
from ECG_analysis import ECG_analysis
from EMG_analysis import emg_analysis
from DS_log_analysis import DS_log_analysis
from smarteye_analysis import Smarteye_analysis
from nasatlx_analysis import Nasatlx_analysis


def main():
    subject_num=2
    delay=48 if subject_num!=1 else 35
    path_to_data_dir=r"/content/drive/MyDrive"
    # experiment_num=1
    categorized_experiment_num=[
        {"num":[1,2,3,4,5],"title":[],"large_title_rri":None,"large_title_fft":None,"large_title_velocity":None,"large_title_distance_of_vehicles":None,"large_title_relative_velocity":None,"large_title_accelerator_and_brake":None,"large_title_steering_torque_and_angle":None,"large_title_gaze_direction":None,"large_title_nasatlx":None,"large_title_mean_beat":None,"before_time":10,"after_time":50},
        {"num":[2,3,4,5],"title":["バスの加速度:0.05g","バスの加速度:0.1g","バスの加速度:0.15g","バスの加速度:0.2g"],"large_title_rri":"RRIの変化（自車：直進，バス：直進で信号が青のうちから減速するとき）","large_title_fft":"RRIの周波数領域図（自車：直進，バス：直進で信号が青のうちから減速するとき）","large_title_velocity":"速さ（自車：直進，バス：直進で信号が青のうちから減速するとき）","large_title_distance_of_vehicles":"車間距離（自車：直進，バス：直進で信号が青のうちから減速するとき）","large_title_relative_velocity":"相対速度（自車：直進，バス：直進で信号が青のうちから減速するとき）","large_title_accelerator_and_brake":"アクセルとブレーキの踏み込み具合（自車：直進，バス：直進で信号が青のうちから減速するとき）","large_title_steering_torque_and_angle":"ステアリングトルクとステアリング角（自車：直進，バス：直進で信号が青のうちから減速するとき）","large_title_gaze_direction":"視線の方向（自車：直進，バス：直進で信号が青のうちから減速するとき）","large_title_nasatlx":"nasa-tlxによるワークロード（自車：直進，バス：直進で信号が青のうちから減速するとき）","large_title_mean_beat":"平均心拍数（自車：直進，バス：直進で信号が青のうちから減速するとき）","before_time":10,"after_time":50},
        {"num":[6,7,8],"title":["バスの向心加速度:0.1g","バスの向心加速度:0.2g","バスの向心加速度:0.3g"],"large_title_rri":"RRIの変化（自車：左折，バス：左折のとき）","large_title_fft":"RRIの周波数領域図（自車：左折，バス：左折のとき）","large_title_velocity":"速さ（自車：左折，バス：左折のとき）","large_title_distance_of_vehicles":"車間距離（自車：左折，バス：左折のとき）","large_title_relative_velocity":"相対速度（自車：左折，バス：左折のとき）","large_title_accelerator_and_brake":"アクセルとブレーキの踏み込み具合（自車：左折，バス：左折のとき）","large_title_steering_torque_and_angle":"ステアリングトルクとステアリング角（自車：左折，バス：左折のとき）","large_title_gaze_direction":"視線の方向（自車：左折，バス：左折のとき）","large_title_nasatlx":"nasa-tlxによるワークロード（自車：左折，バス：左折のとき）","large_title_mean_beat":"平均心拍数（自車：左折，バス：左折のとき）","before_time":10,"after_time":30},
        {"num":[9,10,11],"title":["バスの向心加速度:0.1g","バスの向心加速度:0.2g","バスの向心加速度:0.3g"],"large_title_rri":"RRIの変化（自車：直進，バス：左折のとき）","large_title_fft":"RRIの周波数領域図（自車：直進，バス：左折のとき）","large_title_velocity":"速さ（自車：直進，バス：左折のとき）","large_title_distance_of_vehicles":"車間距離（自車：直進，バス：左折のとき）","large_title_relative_velocity":"相対速度（自車：直進，バス：左折のとき）","large_title_accelerator_and_brake":"アクセルとブレーキの踏み込み具合（自車：直進，バス：左折のとき）","large_title_steering_torque_and_angle":"ステアリングトルクとステアリング角（自車：直進，バス：左折のとき）","large_title_gaze_direction":"視線の方向（自車：直進，バス：左折のとき）","large_title_nasatlx":"nasa-tlxによるワークロード（自車：直進，バス：左折のとき）","large_title_mean_beat":"平均心拍数（自車：直進，バス：左折のとき）","before_time":10,"after_time":30},
        {"num":[12,13,14],"title":["バスの向心加速度:0.1g","バスの向心加速度:0.2g","バスの向心加速度:0.3g"],"large_title_rri":"RRIの変化（自車：直進，バス：右折のとき）","large_title_fft":"RRIの周波数領域図（自車：直進，バス：右折のとき）","large_title_velocity":"速さ（自車：直進，バス：右折のとき）","large_title_distance_of_vehicles":"車間距離（自車：直進，バス：右折のとき）","large_title_relative_velocity":"相対速度（自車：直進，バス：右折のとき）","large_title_accelerator_and_brake":"アクセルとブレーキの踏み込み具合（自車：直進，バス：右折のとき）","large_title_steering_torque_and_angle":"ステアリングトルクとステアリング角（自車：直進，バス：右折のとき）","large_title_gaze_direction":"視線の方向（自車：直進，バス：右折のとき）","large_title_nasatlx":"nasa-tlxによるワークロード（自車：直進，バス：右折のとき）","large_title_mean_beat":"平均心拍数（自車：直進，バス：右折のとき）","before_time":10,"after_time":30},
        {"num":[15,16,17],"title":["バスの向心加速度:0.1g","バスの向心加速度:0.2g","バスの向心加速度:0.3g"],"large_title_rri":"RRIの変化（自車：右折，バス：右折のとき）","large_title_fft":"RRIの周波数領域図（自車：右折，バス：右折のとき）","large_title_velocity":"速さ（自車：右折，バス：右折のとき）","large_title_distance_of_vehicles":"車間距離（自車：右折，バス：右折のとき）","large_title_relative_velocity":"相対速度（自車：右折，バス：右折のとき）","large_title_accelerator_and_brake":"アクセルとブレーキの踏み込み具合（自車：右折，バス：右折のとき）","large_title_steering_torque_and_angle":"ステアリングトルクとステアリング角（自車：右折，バス：右折のとき）","large_title_gaze_direction":"視線の方向（自車：右折，バス：右折のとき）","large_title_nasatlx":"nasa-tlxによるワークロード（自車：右折，バス：右折のとき）","large_title_mean_beat":"平均心拍数（自車：右折，バス：右折のとき）","before_time":10,"after_time":30},
        {"num":[18,19,20],"title":["バスが右折するときの対向車の間隔:160m","バスが右折するときの対向車の間隔:200m","バスが右折するときの対向車の間隔:240m"],"large_title_rri":"RRIの変化（自車：直進，バス：右折のとき）","large_title_fft":"RRIの周波数領域図（自車：直進，バス：右折のとき）","large_title_velocity":"速さ（自車：直進，バス：右折のとき）","large_title_distance_of_vehicles":"車間距離（自車：直進，バス：右折のとき）","large_title_relative_velocity":"相対速度（自車：直進，バス：右折のとき）","large_title_accelerator_and_brake":"アクセルとブレーキの踏み込み具合（自車：直進，バス：右折のとき）","large_title_steering_torque_and_angle":"ステアリングトルクとステアリング角（自車：直進，バス：右折のとき）","large_title_gaze_direction":"視線の方向（自車：直進，バス：右折のとき）","large_title_nasatlx":"nasa-tlxによるワークロード（自車：直進，バス：右折のとき）","large_title_mean_beat":"平均心拍数（自車：直進，バス：右折のとき）","before_time":10,"after_time":30},
        {"num":[21,22,23],"title":["バスが右折するときの対向車の間隔:160m","バスが右折するときの対向車の間隔:200m","バスが右折するときの対向車の間隔:240m"],"large_title_rri":"RRIの変化（自車：右折，バス：右折のとき）","large_title_fft":"RRIの周波数領域図（自車：右折，バス：右折のとき）","large_title_velocity":"速さ（自車：右折，バス：右折のとき）","large_title_distance_of_vehicles":"車間距離（自車：右折，バス：右折のとき）","large_title_relative_velocity":"相対速度（自車：右折，バス：右折のとき）","large_title_accelerator_and_brake":"アクセルとブレーキの踏み込み具合（自車：右折，バス：右折のとき）","large_title_steering_torque_and_angle":"ステアリングトルクとステアリング角（自車：右折，バス：右折のとき）","large_title_gaze_direction":"視線の方向（自車：右折，バス：右折のとき）","large_title_nasatlx":"nasa-tlxによるワークロード（自車：右折，バス：右折のとき）","large_title_mean_beat":"平均心拍数（自車：右折，バス：右折のとき）","before_time":10,"after_time":30}
        ]
    experiment_info=[
        {"before_time":int(5228/120)-35,"after_time":50},
        {"before_time":int(7052/120)-40,"after_time":50},
        {"before_time":int(6856/120)-40,"after_time":50},
        {"before_time":int(6848/120)-40,"after_time":50},
        {"before_time":int(6850/120)-40,"after_time":50},
        {"before_time":int(2886/120)-10,"after_time":30},
        {"before_time":int(2850/120)-10,"after_time":30},
        {"before_time":int(2889/120)-10,"after_time":30},
        {"before_time":int(2886/120)-10,"after_time":30},
        {"before_time":int(2850/120)-10,"after_time":30},
        {"before_time":int(2889/120)-10,"after_time":30},
        {"before_time":int(2876/120)-10,"after_time":30},
        {"before_time":int(2822/120)-10,"after_time":30},
        {"before_time":int(2802/120)-10,"after_time":30},
        {"before_time":int(2876/120)-10,"after_time":30},
        {"before_time":int(2822/120)-10,"after_time":30},
        {"before_time":int(2802/120)-10,"after_time":30},
        {"before_time":int(2508/120)-10,"after_time":45},
        {"before_time":int(2508/120)-10,"after_time":45},
        {"before_time":int(2508/120)-10,"after_time":45},
        {"before_time":int(2508/120)-10,"after_time":45},
        {"before_time":int(2508/120)-10,"after_time":45},
        {"before_time":int(2508/120)-10,"after_time":45}
        ]
    
    
    dfs_list=[]

    for experiment_num in range(1,24):
        print(experiment_num)
        obj=Time_fitting(experiment_num=experiment_num,subject_num=subject_num,before_time=experiment_info[experiment_num-1]["before_time"],after_time=experiment_info[experiment_num-1]["after_time"],delay=delay,path_to_data_dir=path_to_data_dir)
        polymate_df=obj.time_fitting_polymate(check_delay=False)
        smarteye_df=obj.time_fitting_smarteye()
        DS_log_df=obj.time_fitting_DS_log()
        dfs_list.append([polymate_df, smarteye_df, DS_log_df])
        obj.cut_video()
        ECG_analysis(subject_num=subject_num,experiment_num=experiment_num,ecg_df=polymate_df,path_to_data_dir=path_to_data_dir).show_single(False,True,True)
        emg_res=emg_analysis(polymate_df,experiment_num=experiment_num,subject_num=subject_num,show=False,path_to_data_dir=path_to_data_dir)
        DS_log_analysis(experiment_num=experiment_num,subject_num=subject_num,df=DS_log_df,path_to_data_dir=path_to_data_dir).analyze_all(show=False)
        Smarteye_analysis(subject_num=subject_num,experiment_num=experiment_num,smarteye_df=smarteye_df,path_to_data_dir=path_to_data_dir).show_gaze_direction(show=False)
    
    for experiment_nums in categorized_experiment_num:
        polymate_dfs=[]
        smarteye_dfs=[]
        DS_log_dfs=[]
        for experiment_num in experiment_nums["num"]:
            polymate_dfs.append(dfs_list[experiment_num-1][0])
            smarteye_dfs.append(dfs_list[experiment_num-1][1])
            DS_log_dfs.append(dfs_list[experiment_num-1][2])
        ECG_analysis(subject_num=subject_num,experiment_nums=experiment_nums["num"],ecg_dfs=polymate_dfs,path_to_data_dir=path_to_data_dir).show_multiple(titles=experiment_nums["title"],large_title_rri=experiment_nums["large_title_rri"],large_title_fft=experiment_nums["large_title_fft"],large_title_mean_beat=experiment_nums["large_title_mean_beat"],show=False,store=True,modify=True)
        DS_log_analysis(subject_num=subject_num,experiment_nums=experiment_nums["num"],dfs=DS_log_dfs,path_to_data_dir=path_to_data_dir).multi_analyze_all(titles=experiment_nums["title"],large_title_distance_of_vehicles=experiment_nums["large_title_distance_of_vehicles"],large_title_relative_velocity=experiment_nums["large_title_relative_velocity"],large_title_accelerator_and_brake=experiment_nums["large_title_accelerator_and_brake"],large_title_steering_torque_and_angle=experiment_nums["large_title_steering_torque_and_angle"],store=True,show=False)
        Smarteye_analysis(subject_num=subject_num,experiment_nums=experiment_nums["num"],smarteye_dfs=smarteye_dfs,path_to_data_dir=path_to_data_dir).multi_analyze(titles=experiment_nums["title"],large_title_gaze_direction=experiment_nums["large_title_gaze_direction"],store=True,show=False)
        Nasatlx_analysis(subject_num=subject_num,path_to_data_dir=path_to_data_dir).show_workload_frustration(experiment_nums=experiment_nums["num"],titles=experiment_nums["title"],large_title=experiment_nums["large_title_nasatlx"],show=False,store=True)
        
if __name__ == '__main__':
    main()