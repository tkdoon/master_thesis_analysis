import os
import pandas as pd
import numpy as np

def rename_smarteye(folder_path,experiment_num_list):
    files = sorted(os.listdir(folder_path), key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
  # ファイルを1から順番にリネーム
    for index, file_name in enumerate(files):
        original_path = os.path.join(folder_path, file_name)
        new_file_name = f"{experiment_num_list[index]}.log"

        # リネーム
        new_path = os.path.join(folder_path, new_file_name)
        os.rename(original_path, new_path)


def load_excel(excel_path):
    # Excelファイルを読み込み
    df = pd.read_excel(excel_path, header=None)

    # DataFrameをNumPyの行列に変換
    matrix = df.values
    return matrix





if __name__=="__main__":
    subject_num=12
    dir_path=fr"D:\修士研究\実験データ\{subject_num}\smarteye"
    experiment_num_path=r"ラテン方格法.xlsm"
    experiment_num_list=load_excel(experiment_num_path)[subject_num-1,:].astype(int)
    rename_smarteye(dir_path,experiment_num_list)