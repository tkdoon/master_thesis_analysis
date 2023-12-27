import numpy as np
import pandas as pd
from convert_from_DS_index import search_index_in_DSlog,DS_log_index2second

categorized_experiment_num=[
        {
            "num":[1,2,3,4,5],
            "title":[],
            "large_title_rri":None,
            "large_title_fft":None,
            "large_title_velocity":None,
            "large_title_distance_of_vehicles":None,
            "large_title_relative_velocity":None,
            "large_title_accelerator_and_brake":None,
            "large_title_steering_torque_and_angle":None,
            "large_title_gaze_direction":None,
            "large_title_nasatlx":None,
            "large_title_mean_beat":None,
            "large_title_acceleration":None,
            "large_title_steering_angle":None,
            "large_title_yaw":None,
            "large_title_emg":None},
        {
            "num":[2,3,4,5],
            "title":["バスの加速度:0.05G","バスの加速度:0.1G","バスの加速度:0.15G","バスの加速度:0.2G"],
            "large_title_rri":"RRIの変化（自車：直進，バス：直進で信号が青のうちから減速するとき）",
            "large_title_fft":"RRIの周波数領域図（自車：直進，バス：直進で信号が青のうちから減速するとき）",
            "large_title_velocity":"速さ（自車：直進，バス：直進で信号が青のうちから減速するとき）",
            "large_title_distance_of_vehicles":"車間距離（自車：直進，バス：直進で信号が青のうちから減速するとき）",
            "large_title_relative_velocity":"相対速度（自車：直進，バス：直進で信号が青のうちから減速するとき）",
            "large_title_accelerator_and_brake":"アクセルとブレーキの踏み込み具合（自車：直進，バス：直進で信号が青のうちから減速するとき）",
            "large_title_steering_torque_and_angle":"ステアリングトルクとステアリング角（自車：直進，バス：直進で信号が青のうちから減速するとき）",
            "large_title_gaze_direction":"視線の方向（自車：直進，バス：直進で信号が青のうちから減速するとき）",
            "large_title_nasatlx":"nasa-tlxによるワークロード（自車：直進，バス：直進で信号が青のうちから減速するとき）",
            "large_title_mean_beat":"平均心拍数（自車：直進，バス：直進で信号が青のうちから減速するとき）",
            "large_title_acceleration":"加速度の絶対値（自車：直進，バス：直進で信号が青のうちから減速するとき)",
            "large_title_steering_angle":"ステアリング角（自車：直進，バス：直進で信号が青のうちから減速するとき）",
            "large_title_yaw":"ヨー角（自車：直進，バス：直進で信号が青のうちから減速するとき）",
            "large_title_emg":"EMG（自車：直進，バス：直進で信号が青のうちから減速するとき）"},
        {
            "num":[6,7,8],
            "title":["バスの向心加速度:0.1G","バスの向心加速度:0.2G","バスの向心加速度:0.3G"],
            "large_title_rri":"RRIの変化（自車：直進，バス：左折のとき）",
            "large_title_fft":"RRIの周波数領域図（自車：直進，バス：左折のとき）",
            "large_title_velocity":"速さ（自車：直進，バス：左折のとき）",
            "large_title_distance_of_vehicles":"車間距離（自車：直進，バス：左折のとき）",
            "large_title_relative_velocity":"相対速度（自車：直進，バス：左折のとき）",
            "large_title_accelerator_and_brake":"アクセルとブレーキの踏み込み具合（自車：直進，バス：左折のとき）",
            "large_title_steering_torque_and_angle":"ステアリングトルクとステアリング角（自車：直進，バス：左折のとき）",
            "large_title_gaze_direction":"視線の方向（自車：直進，バス：左折のとき）",
            "large_title_nasatlx":"nasa-tlxによるワークロード（自車：直進，バス：左折のとき）",
            "large_title_mean_beat":"平均心拍数（自車：直進，バス：左折のとき）",
            "large_title_acceleration":"加速度の絶対値（自車：直進，バス：左折のとき）",
            "large_title_steering_angle":"ステアリング角（自車：直進，バス：左折のとき）",
            "large_title_yaw":"ヨー角（自車：直進，バス：左折のとき）",
            "large_title_emg":"EMG（自車：直進，バス：左折のとき）"},
        {
            "num":[9,10,11],
            "title":["バスの向心加速度:0.1G","バスの向心加速度:0.2G","バスの向心加速度:0.3G"],
            "large_title_rri":"RRIの変化（自車：左折，バス：左折のとき）",
            "large_title_fft":"RRIの周波数領域図（自車：左折，バス：左折のとき）",
            "large_title_velocity":"速さ（自車：左折，バス：左折のとき）",
            "large_title_distance_of_vehicles":"車間距離（自車：左折，バス：左折のとき）",
            "large_title_relative_velocity":"相対速度（自車：左折，バス：左折のとき）",
            "large_title_accelerator_and_brake":"アクセルとブレーキの踏み込み具合（自車：左折，バス：左折のとき）",
            "large_title_steering_torque_and_angle":"ステアリングトルクとステアリング角（自車：左折，バス：左折のとき）",
            "large_title_gaze_direction":"視線の方向（自車：左折，バス：左折のとき）",
            "large_title_nasatlx":"nasa-tlxによるワークロード（自車：左折，バス：左折のとき）",
            "large_title_mean_beat":"平均心拍数（自車：左折，バス：左折のとき）",
            "large_title_acceleration":"加速度の絶対値（自車：左折，バス：左折のとき）",
            "large_title_steering_angle":"ステアリング角（自車：左折，バス：左折のとき）",
            "large_title_yaw":"ヨー角（自車：左折，バス：左折のとき）",
            "large_title_emg":"EMG（自車：左折，バス：左折のとき）"},
        {
            "num":[12,13,14],
            "title":["バスの向心加速度:0.1G","バスの向心加速度:0.2G","バスの向心加速度:0.3G"],
            "large_title_rri":"RRIの変化（自車：直進，バス：右折のとき）",
            "large_title_fft":"RRIの周波数領域図（自車：直進，バス：右折のとき）",
            "large_title_velocity":"速さ（自車：直進，バス：右折のとき）",
            "large_title_distance_of_vehicles":"車間距離（自車：直進，バス：右折のとき）",
            "large_title_relative_velocity":"相対速度（自車：直進，バス：右折のとき）",
            "large_title_accelerator_and_brake":"アクセルとブレーキの踏み込み具合（自車：直進，バス：右折のとき）",
            "large_title_steering_torque_and_angle":"ステアリングトルクとステアリング角（自車：直進，バス：右折のとき）",
            "large_title_gaze_direction":"視線の方向（自車：直進，バス：右折のとき）",
            "large_title_nasatlx":"nasa-tlxによるワークロード（自車：直進，バス：右折のとき）",
            "large_title_mean_beat":"平均心拍数（自車：直進，バス：右折のとき）",
            "large_title_acceleration":"加速度の絶対値（自車：直進，バス：右折のとき）",
            "large_title_steering_angle":"ステアリング角（自車：直進，バス：右折のとき）",
            "large_title_yaw":"ヨー角（自車：直進，バス：右折のとき）",
            "large_title_emg":"EMG（自車：直進，バス：右折のとき）"},
        {
            "num":[15,16,17],
            "title":["バスの向心加速度:0.1G","バスの向心加速度:0.2G","バスの向心加速度:0.3G"],
            "large_title_rri":"RRIの変化（自車：右折，バス：右折のとき）",
            "large_title_fft":"RRIの周波数領域図（自車：右折，バス：右折のとき）",
            "large_title_velocity":"速さ（自車：右折，バス：右折のとき）",
            "large_title_distance_of_vehicles":"車間距離（自車：右折，バス：右折のとき）",
            "large_title_relative_velocity":"相対速度（自車：右折，バス：右折のとき）",
            "large_title_accelerator_and_brake":"アクセルとブレーキの踏み込み具合（自車：右折，バス：右折のとき）",
            "large_title_steering_torque_and_angle":"ステアリングトルクとステアリング角（自車：右折，バス：右折のとき）",
            "large_title_gaze_direction":"視線の方向（自車：右折，バス：右折のとき）",
            "large_title_nasatlx":"nasa-tlxによるワークロード（自車：右折，バス：右折のとき）",
            "large_title_mean_beat":"平均心拍数（自車：右折，バス：右折のとき）","large_title_acceleration":"加速度の絶対値（自車：右折，バス：右折のとき）","large_title_steering_angle":"ステアリング角（自車：右折，バス：右折のとき）",
            "large_title_yaw":"ヨー角（自車：右折，バス：右折のとき）",
            "large_title_emg":"EMG（自車：右折，バス：右折のとき）"},
        {
            "num":[18,19,20],
            "title":["バスが右折するときの対向車の間隔:160m","バスが右折するときの対向車の間隔:200m","バスが右折するときの対向車の間隔:240m"],
            "large_title_rri":"RRIの変化（自車：直進，バス：右折のとき）",
            "large_title_fft":"RRIの周波数領域図（自車：直進，バス：右折のとき）",
            "large_title_velocity":"速さ（自車：直進，バス：右折のとき）",
            "large_title_distance_of_vehicles":"車間距離（自車：直進，バス：右折のとき）",
            "large_title_relative_velocity":"相対速度（自車：直進，バス：右折のとき）",
            "large_title_accelerator_and_brake":"アクセルとブレーキの踏み込み具合（自車：直進，バス：右折のとき）",
            "large_title_steering_torque_and_angle":"ステアリングトルクとステアリング角（自車：直進，バス：右折のとき）",
            "large_title_gaze_direction":"視線の方向（自車：直進，バス：右折のとき）",
            "large_title_nasatlx":"nasa-tlxによるワークロード（自車：直進，バス：右折のとき）",
            "large_title_mean_beat":"平均心拍数（自車：直進，バス：右折のとき）",
            "large_title_acceleration":"加速度の絶対値（自車：直進，バス：右折のとき）",
            "large_title_steering_angle":"ステアリング角（自車：直進，バス：右折のとき）",
            "large_title_yaw":"ヨー角（自車：直進，バス：右折のとき）",
            "large_title_emg":"EMG（自車：直進，バス：右折のとき）"},
        {
            "num":[21,22,23],
            "title":["バスが右折するときの対向車の間隔:160m","バスが右折するときの対向車の間隔:200m","バスが右折するときの対向車の間隔:240m"],
            "large_title_rri":"RRIの変化（自車：右折，バス：右折のとき）",
            "large_title_fft":"RRIの周波数領域図（自車：右折，バス：右折のとき）",
            "large_title_velocity":"速さ（自車：右折，バス：右折のとき）",
            "large_title_distance_of_vehicles":"車間距離（自車：右折，バス：右折のとき）",
            "large_title_relative_velocity":"相対速度（自車：右折，バス：右折のとき）",
            "large_title_accelerator_and_brake":"アクセルとブレーキの踏み込み具合（自車：右折，バス：右折のとき）",
            "large_title_steering_torque_and_angle":"ステアリングトルクとステアリング角（自車：右折，バス：右折のとき）",
            "large_title_gaze_direction":"視線の方向（自車：右折，バス：右折のとき）",
            "large_title_nasatlx":"nasa-tlxによるワークロード（自車：右折，バス：右折のとき）",
            "large_title_mean_beat":"平均心拍数（自車：右折，バス：右折のとき）",
            "large_title_acceleration":"加速度の絶対値（自車：右折，バス：右折のとき）",
            "large_title_steering_angle":"ステアリング角（自車：右折，バス：右折のとき）",
            "large_title_yaw":"ヨー角（自車：右折，バス：右折のとき）",
            "large_title_emg":"EMG（自車：右折，バス：右折のとき）"}
        ]
# experiment_info=[
#     # 直進
#         {"before_time":int(5228/120)-35,"after_time":50,"delete_zero":True},
#         {"before_time":int(7052/120)-40,"after_time":55,"delete_zero":True},
#         {"before_time":int(6856/120)-40,"after_time":55,"delete_zero":True},
#         {"before_time":int(6848/120)-40,"after_time":55,"delete_zero":True},
#         {"before_time":int(6850/120)-40,"after_time":55,"delete_zero":True},
#         # バス＝左折，自車＝直進
#         {"before_time":int(2886/120)-10,"after_time":25,"delete_zero":False},
#         {"before_time":int(2850/120)-10,"after_time":25,"delete_zero":False},
#         {"before_time":int(2889/120)-10,"after_time":25,"delete_zero":False},
#         # バス=左折，自車＝左折
#         {"before_time":int(2886/120)-10,"after_time":30,"delete_zero":False},
#         {"before_time":int(2850/120)-10,"after_time":30,"delete_zero":False},
#         {"before_time":int(2889/120)-10,"after_time":30,"delete_zero":False},
#         # バス=右折，自車＝直進
#         {"before_time":int(2876/120)-10,"after_time":25,"delete_zero":False},
#         {"before_time":int(2822/120)-10,"after_time":25,"delete_zero":False},
#         {"before_time":int(2802/120)-10,"after_time":25,"delete_zero":False},
#         # バス＝右折，自車＝右折
#         {"before_time":int(2876/120)-10,"after_time":30,"delete_zero":False},
#         {"before_time":int(2822/120)-10,"after_time":30,"delete_zero":False},
#         {"before_time":int(2802/120)-10,"after_time":30,"delete_zero":False},
#         # バス＝右折，自車=直進，対向車
#         {"before_time":int(2508/120)-10,"after_time":45,"delete_zero":True},
#         {"before_time":int(2508/120)-10,"after_time":50,"delete_zero":True},
#         {"before_time":int(2508/120)-10,"after_time":55,"delete_zero":True},
#         # バス＝右折，自車＝右折，対向車
#         {"before_time":int(2508/120)-10,"after_time":45,"delete_zero":True},
#         {"before_time":int(2508/120)-10,"after_time":50,"delete_zero":True},
#         {"before_time":int(2508/120)-10,"after_time":55,"delete_zero":True}
#         ]


def create_experiment_info(DS_log_df:pd.DataFrame,experiment_num:int)->dict:
    
    experiment_info=[
        # 直進（バスが減速を始めるところから，バスが加速し終わって3秒後まで）
        {"before_search_for":"speed down start","after_search_for":"speed up end","after_time_after_event":3,"delete_zero":True,"emg":[{"start":"speed down start","end":"speed down end"},{"start":"speed up start","end":"speed up end"}]},
        {"before_search_for":"speed down start","after_search_for":"speed up end","after_time_after_event":3,"delete_zero":True,"emg":[{"start":"speed down start","end":"speed down end"},{"start":"speed up start","end":"speed up end"}]},
        {"before_search_for":"speed down start","after_search_for":"speed up end","after_time_after_event":3,"delete_zero":True,"emg":[{"start":"speed down start","end":"speed down end"},{"start":"speed up start","end":"speed up end"}]},
        {"before_search_for":"speed down start","after_search_for":"speed up end","after_time_after_event":3,"delete_zero":True,"emg":[{"start":"speed down start","end":"speed down end"},{"start":"speed up start","end":"speed up end"}]},
        {"before_search_for":"speed down start","after_search_for":"speed up end","after_time_after_event":3,"delete_zero":True,"emg":[{"start":"speed down start","end":"speed down end"},{"start":"speed up start","end":"speed up end"}]},

        # バス＝左折，自車＝直進（バスが減速を開始するところから，交差点通過後5秒）
        {"before_search_for":"speed down start","after_search_for":"speed up start","after_time_after_event":5,"delete_zero":False,"emg":[{"start":"speed down end","end":"speed up start"}]},
        {"before_search_for":"speed down start","after_search_for":"speed up start","after_time_after_event":5,"delete_zero":False,"emg":[{"start":"speed down end","end":"speed up start"}]},
        {"before_search_for":"speed down start","after_search_for":"speed up start","after_time_after_event":5,"delete_zero":False,"emg":[{"start":"speed down end","end":"speed up start"}]},

        # バス=左折，自車＝左折（バスが減速を開始するところから，左折後加速し終わった後3秒）
        {"before_search_for":"speed down start","after_search_for":"speed up end","after_time_after_event":3,"delete_zero":False,"emg":[{"start":"speed down end","end":"speed up start"}]},
        {"before_search_for":"speed down start","after_search_for":"speed up end","after_time_after_event":3,"delete_zero":False,"emg":[{"start":"speed down end","end":"speed up start"}]},
        {"before_search_for":"speed down start","after_search_for":"speed up end","after_time_after_event":3,"delete_zero":False,"emg":[{"start":"speed down end","end":"speed up start"}]},

        # バス=右折，自車＝直進（バスが減速を開始するところから，交差点通過後5秒）
        {"before_search_for":"speed down start","after_search_for":"speed up start","after_time_after_event":5,"delete_zero":False,"emg":[{"start":"speed down end","end":"speed up start"}]},
        {"before_search_for":"speed down start","after_search_for":"speed up start","after_time_after_event":5,"delete_zero":False,"emg":[{"start":"speed down end","end":"speed up start"}]},
        {"before_search_for":"speed down start","after_search_for":"speed up start","after_time_after_event":5,"delete_zero":False,"emg":[{"start":"speed down end","end":"speed up start"}]},
        # バス＝右折，自車＝右折（バスが減速を開始するところから，右折後加速し終わった後3秒）
        {"before_search_for":"speed down start","after_search_for":"speed up end","after_time_after_event":3,"delete_zero":False,"emg":[{"start":"speed down end","end":"speed up start"}]},
        {"before_search_for":"speed down start","after_search_for":"speed up end","after_time_after_event":3,"delete_zero":False,"emg":[{"start":"speed down end","end":"speed up start"}]},
        {"before_search_for":"speed down start","after_search_for":"speed up end","after_time_after_event":3,"delete_zero":False,"emg":[{"start":"speed down end","end":"speed up start"}]},
        # バス＝右折，自車=直進，対向車（バスが減速を開始するところから，交差点通過後5秒）
        {"before_search_for":"speed down start","after_search_for":"speed up start","after_time_after_event":5,"delete_zero":False,"emg":[{"start":"speed up start","end":"speed up end"}]},
        {"before_search_for":"speed down start","after_search_for":"speed up start","after_time_after_event":5,"delete_zero":False,"emg":[{"start":"speed up start","end":"speed up end"}]},
        {"before_search_for":"speed down start","after_search_for":"speed up start","after_time_after_event":5,"delete_zero":False,"emg":[{"start":"speed up start","end":"speed up end"}]},
        # バス＝右折，自車＝右折，対向車（バスが減速を開始するところから，右折後加速し終わった後3秒）
        {"before_search_for":"speed down start","after_search_for":"speed up end","after_time_after_event":3,"delete_zero":False,"emg":[{"start":"speed up start","end":"speed up end"}]},
        {"before_search_for":"speed down start","after_search_for":"speed up end","after_time_after_event":3,"delete_zero":False,"emg":[{"start":"speed up start","end":"speed up end"}]},
        {"before_search_for":"speed down start","after_search_for":"speed up end","after_time_after_event":3,"delete_zero":False,"emg":[{"start":"speed up start","end":"speed up end"}]},
        ]

    before_time_index=search_index_in_DSlog(DS_log_df,experiment_info[experiment_num-1]["before_search_for"])
    before_time=DS_log_index2second(DS_log_df,before_time_index)
    after_time_index=search_index_in_DSlog(DS_log_df,experiment_info[experiment_num-1]["after_search_for"])
    after_time=DS_log_index2second(DS_log_df,after_time_index)+experiment_info[experiment_num-1]["after_time_after_event"]
    emg_time=[]
    for  start_end in experiment_info[experiment_num-1]["emg"]:
        emg_before_time_index=search_index_in_DSlog(DS_log_df,start_end["start"])
        emg_before_time=DS_log_index2second(DS_log_df,emg_before_time_index)
        emg_after_time_index=search_index_in_DSlog(DS_log_df,start_end["end"])
        emg_after_time=DS_log_index2second(DS_log_df,emg_after_time_index)
        emg_time.append({"before_time":emg_before_time, "after_time":emg_after_time})
    
    return {"before_time":before_time,"after_time":after_time,"delete_zero":experiment_info[experiment_num-1]["delete_zero"],"emg":emg_time}

