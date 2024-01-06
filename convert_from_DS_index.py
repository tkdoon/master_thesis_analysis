import pandas as pd
import numpy as np
from time_fitting import Time_fitting
from datetime import datetime,timedelta
from typing import Union

def search_index_in_DSlog(DS_log_df:pd.DataFrame,search_for:str):
    """search forは'speed down start' or 'speed up end' or 'pass section end' or 'speed down end' or 'speed up start' or 'your car not stop',your car not stopのみindexのnumpy配列を返し，それ以外はindexのintを返す"""
    DS_log_np=DS_log_df.values
    bus_velocity_array=DS_log_np[:,42]
    bus_direction_array=DS_log_np[:,41]
    max_velocity=np.max(bus_velocity_array)#巡航速度
    min_velocity=np.min(bus_velocity_array)
    speed_down_start_index_array=np.where(bus_velocity_array[1:]<bus_velocity_array[:-1])[0]+1#値が小さくなるインデックス
    for speed_down_start_index in speed_down_start_index_array:
        next_max_speed_index=speed_down_start_index+np.where(bus_velocity_array[speed_down_start_index:]==max_velocity)[0][0]
        if(len(np.where(bus_velocity_array[speed_down_start_index:next_max_speed_index]<10)[0])!=0):
            break
    
    if search_for=="speed down start":
        return speed_down_start_index
    elif search_for=="speed down end":
        min_velocity=np.min(bus_velocity_array[speed_down_start_index:])
        min_speed_index=np.where(bus_velocity_array[speed_down_start_index:]==min_velocity)[0]+speed_down_start_index
        speed_down_end_index=min_speed_index[0]
        return speed_down_end_index
    elif search_for=="speed up start":
        bus_velocity_array_after_speed_down=bus_velocity_array[speed_down_start_index:]
        speed_up_start_index=speed_down_start_index+np.where(bus_velocity_array_after_speed_down[1:]>bus_velocity_array_after_speed_down[:-1])[0][0]+1
        return speed_up_start_index
    elif search_for=="speed up end":
        # speed_down_index=np.where(bus_velocity_array[1:]<bus_velocity_array[:-1])[0][0]+1
        speed_up_end_index=speed_down_start_index+np.where(bus_velocity_array[speed_down_start_index:]==max_velocity)[0][0]
        return speed_up_end_index
    elif search_for=="pass intersection end":
        pass_end_index=np.where(np.logical_or(bus_direction_array>=3.09,bus_direction_array<=-0.038))[0][0]
        return pass_end_index
    elif search_for=="your car stop":
        your_car_velocity_array=DS_log_np[:,34]
        # 値が0の場所のインデックスを抜き出す
        zero_indices = np.where(your_car_velocity_array==0)[0]
        print(zero_indices.shape)
        return zero_indices
    elif search_for=="other car1 passed":
        return 4604
    elif search_for=="other car2 passed":
        return 6084
    elif search_for=="other car3 passed":
        return 7828
    else:
        return None

def DS_log_index2second(DS_log_df:pd.DataFrame,index:int):
    DS_log_np=DS_log_df.values
    time_array=DS_log_np[:,11]
    return time_array[index]

    
def convert_index_list(index_array):
    # [0,1,2,3,4,6,7,8,9]->[[0,4],[6,9]]
    # 配列の差分を計算し、連続する部分を見つける
    diff = np.diff(index_array)
    split_indices = np.where(diff != 1)[0] + 1

    # 連続した範囲を抽出し、各範囲の最初と最後の値を取得
    ranges = np.split(index_array, split_indices)
    result = [[sub_arr[0], sub_arr[-1]] for sub_arr in ranges if len(sub_arr) > 0]
    return np.array(result)


    
    
    
def search_seconds_in_DS_log(DS_log_df:pd.DataFrame,search_for:str):
    index=search_index_in_DSlog(DS_log_df,search_for)
    if(type(index)==int):
        return DS_log_index2second(DS_log_df,index)
    else:
        new_index_array=convert_index_list(index)
        seconds_list=[]
        for sub_array in new_index_array:
            seconds_list.append([DS_log_index2second(DS_log_df,sub_array[0]),DS_log_index2second(DS_log_df,sub_array[1])])
        seconds_array=np.array(seconds_list)
        return seconds_array
    
    



if __name__ == '__main__':
    # res=make_smarteye_seconds_list(0,0,np.array([0,1,2,3,4,6,7,8,9,34,21,22]))
    print("")
    