from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.stats.anova as anova
import pandas as pd
from itertools import combinations

def shapiro_wilk(alpha,data:list):
    """_summary_
    正規性が認められればTrue，認められなければFalseを返す．
    Args:
        alpha (_type_): _description_
        data (list): _description_

    Returns:
        _type_: _description_
    """
    # シャピロ-ウィルク検定の実行
    statistic, p_value = stats.shapiro(data)

    return p_value < alpha

def QQplot(data):
    # 正規QQプロットの作成
    probplot_data = stats.probplot(data, plot=plt)

    # グリッド線の追加（オプション）
    plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)
    # グラフの表示
    plt.title('Normal QQ Plot')
    plt.show()
    
def paired_t_test(alpha,data1,data2):
    """_summary_
    統計的な有意差が認められればTrue，認められなければFalseを返す．
    Args:
        alpha (_type_): _description_
        data1 (_type_): _description_
        data2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    t_statistic,p_value=stats.ttest_rel(data1, data2)
    return p_value < alpha,p_value

def wilcoxon(alpha,data1,data2):
    """_summary_
    統計的な有意差が認められればTrue，認められなければFalseを返す．
    Args:
        alpha (_type_): _description_
        data1 (_type_): _description_
        data2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    t_statistic,p_value=stats.wilcoxon(data1, data2,alternative="two-sided")
    return p_value < alpha,p_value

def repeated_anova(*data_list):
    tdata=pd.DataFrame(data_list)
    aov=anova.AnovaRM(tdata)
    aov.fit()
    print(result)
    
    
def kruskal_wolis(alpha,*data_list):
    """_summary_
    統計的な有意差が認められればTrue，認められなければFalseを返す．
    Args:
        alpha (_type_): _description_

    Returns:
        _type_: _description_
    """
    t_statistic,p_value=stats.kruskal(*data_list)
    return p_value < alpha,p_value    
    

analyze_list=[
    
]

compare_list=[
    [1,2],[2,3,4,5],[6,7,8],[9,10,11],[12,13,14],[15,16,17],[18,19,20],[21,22,23]
]

def main():
    df_list=[]
    for i in range(1,24):
        # path=f{i}
        df=read_df(path)
        df_list.append(df)
    
    
    for analyze_item in analyze_list:
        data_list=[]
        for df1 in df_list:
            data=df1.loc[analyze_item].values
            data_list.append(data)
        # np.array(data_list)
        res=[]
        for compare_item in compare_list:
            prepared_list=[]
            for experiment_num in compare_item:
                temp_list=[]
                for subject_num in  range(23):
                    temp_list.append(data_list[subject_num][experiment_num-1])
                prepared_list.append(temp_list)
            res.append(statistical_analysis(*prepared_list))
            
        # write_path=f{analyze_item}    
        write_csv(write_path,*res)
                
                    
def statistical_analysis(*data_list):
    for data in data_list:
        print("シャピロウィルク:",shapiro_wilk(data))
        QQplot(data)
        judge=input()
    
    if judge=="n":
       normality=False
    elif judge=="y":
        normality=True     
        
    if(len(data_list)==2 and normality):
        difference,p_value=paired_t_test(0.05, data_list[0],data_list[1])
    elif(len(data_list)==2 and not normality):
        difference,p_value=wilcoxon(0.05, data_list[0], data_list[1])
    elif(len(data_list)>=3 and normality):
        
    elif(len(data_list)>=3 and not normality):
        difference,p_value=kruskal_wolis(0.05, *data_list)    
        if (difference):
            combinations_list = list(combinations(data_list, 2))
            res=[]
            for combi in combinations_list:
                each_difference,each_p_value=wilcoxon(0.05/(len(data_list)-1),combi[0],combi[1])
                res.append((combi,each_difference,each_p_value))
    else:
        return 
        

    
    
    
    
    
    
def read_df(path):
    df = pd.read_csv(pd,header=1 )
    return df

def write_csv(path,*args):
    """可変長引数の中身はkey,value1,value2,...の形で渡してください"""
    import csv
    with open(path,'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # データをCSVファイルに書き込む
        csv_writer.writerows(args)


if __name__ == '__main__':
    main()