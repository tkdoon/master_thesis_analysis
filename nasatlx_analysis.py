import matplotlib.pyplot as plt
import numpy as np
import json
import os
import japanize_matplotlib

class Nasatlx_analysis():
    def __init__(self,subject_num:int,path_to_data_dir=r"C:\Users\tyasu\Desktop\修士研究用") -> None:
        self.subject_num=subject_num
        self.nasatlx_data=None
        self.path_to_data_dir=path_to_data_dir
        self.json_file_path=os.path.join(path_to_data_dir,rf"実験データ\{subject_num}\{subject_num}.json")
        
    
    def open_file(self):
        with open(self.json_file_path,encoding="utf-8") as f:
            nasatlx_res=json.load(f)
        return nasatlx_res
    
    
    
    def show_workload_frustration(self,experiment_nums:list[int],titles:list[str],large_title:str,show:bool=True,store:bool=True,font_size:int=12):
        dictionary=self.open_file()
        frustration_list=[]
        overall_workload_list=[]
        filename=""
        for experiment_num in experiment_nums:
            result=json.loads(dictionary[f"experiment{experiment_num}"])
            frustration=int(result["calculatedResult"]["results_rating"][5])
            overall_workload=float(result["calculatedResult"]["results_overall"])
            frustration_list.append(frustration)
            overall_workload_list.append(overall_workload)
            filename+=str(experiment_num)
            
            
        # サンプルデータ
        categories = titles
        data1 = frustration_list
        data2 = overall_workload_list

        # グラフの幅
        bar_width = 0.35
        
        if(len(categories)==0):
            categories=experiment_nums

        # インデックス
        indices = np.arange(len(categories))
        plt.rcParams["font.size"] = font_size
        # データ1の棒グラフ
        plt.bar(indices, data1, bar_width, label='frustrationの値')

        # データ2の棒グラフ。インデックスにバーの幅を足して横に並べる
        plt.bar(indices + bar_width, data2, bar_width, label='ワークロード全体')

        # カテゴリ名を設定
        plt.xticks(indices + bar_width / 2, categories)

        # 凡例を表示
        plt.legend()
        if(large_title):
        # グラフのタイトルと軸ラベル
            plt.title(large_title)
        plt.tight_layout()


        if store:
            if not os.path.exists(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}")):
                os.makedirs(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}"))
            if not os.path.exists(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\nasatlx")):
                os.makedirs(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\nasatlx"))
            plt.savefig(os.path.join(self.path_to_data_dir,fr"解析データ\{self.subject_num}\nasatlx\frustration_workload_{filename}.png"))
        if show:
            plt.show()
        else:
            plt.close()
        
        
        
if __name__=='__main__':
    Nasatlx_analysis(1).show_workload_frustration([1,2,3,4,5])